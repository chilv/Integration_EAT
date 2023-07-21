import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
import statistics
from collections import deque

import numpy as np
import pandas as pd
import yaml
from legged_gym.envs import *
from legged_gym.utils import (Logger, export_policy_as_jit, get_args,
                              task_registry)
from model import LeggedTransformerBody, LeggedTransformerPro
from scripts.utils import flaw_generation

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR

import torch
ENV_NUMS = 512

def test_EAT(args, env, EAT_model, faulty_tag = -1, flawed_rate = 1, pred_body = True):
    # loading EAT model
    eval_batch_size = ENV_NUMS        # 测试环境数
    max_test_ep_len = 1000    	#iters

    body_dim = args["body_dim"]
    if type(faulty_tag) == int:
        tag_list = [faulty_tag]
    else:
        tag_list = faulty_tag
    body_target, _ = flaw_generation(eval_batch_size, body_dim, tag_list, flawed_rate, device)
    # body_target = [1 for _ in range(body_dim)]
    # if not pred_body:
    #     body_target [faulty_tag] = flawed_rate
    # if (faulty_tag != -1):
        # body_target[faulty_tag] = flawed_rate

    state_mean = torch.from_numpy(args["state_mean"]).to(device)
    state_std = torch.from_numpy(args["state_std"]).to(device)
    
    # body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
    # bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32)
    bodies = body_target.unsqueeze(1).repeat(1, max_test_ep_len, 1).type(torch.float32)
    results = {}
    #testing loop
    running_state = env.reset()[0]
    running_reward = torch.zeros((eval_batch_size, ), dtype=torch.float32, device=device)
    
    # timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    # timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
    actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
    states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                        dtype=torch.float32, device=device)
    total_rewards = np.zeros(eval_batch_size)
    total_length = np.zeros(eval_batch_size)
    dones = np.zeros(eval_batch_size)
    
    #testing 
    with torch.no_grad():
        print(f"joint {faulty_tag} with flawed rate {flawed_rate} is under testing")
        for t in range(max_test_ep_len):

            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            if t < context_len:
                _, act_preds, _ = EAT_model.forward(
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            bodies=bodies[:,:context_len])
                act = act_preds[:, t].detach()
            else:
                _, act_preds, _ = EAT_model.forward(
                                        states[:,t-context_len+1:t+1],
                                        actions[:,t-context_len+1:t+1],
                                        bodies=bodies[:,t-context_len+1:t+1])
                act = act_preds[:, -1].detach()
            
            # body, _ = flaw_generation(ENV_NUMS, fixed_joint = [faulty_tag], flawed_rate = flawed_rate, device = )
            running_state, _, running_reward, done, infos = env.step(act, body_target)
            # if t < max_test_ep_len/8:
                # running_state, _, running_reward, done, infos = env.step(act, [-1])
        
            actions[:, t] = act

            # total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            total_length += (dones == 0)
            dones += done.detach().cpu().numpy()

            if np.all(dones):
                break
    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_length)
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")
    
    return results['eval/avg_reward'], results['eval/avg_ep_len']


if __name__ == "__main__":
    with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)

    device = torch.device(args["device"])  # setting flexible
    
    env_args = get_args()
    env_args.sim_device = args["device"]
    env_args.task = args["task"]
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUMS)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.7]# 更改速度设置以防命令采样到0的情况    
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    # # Faster test
    # env_cfg.commands.ranges.lin_vel_x = [-0.7,0.7]
    # env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1,1]
    # prepare environment
    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)
    model_name = "EAT_Given_body_AMP_Teacher_Position_Encoding_AMPPPO_04"
    run_name = "EAT_runs_AMP"
    model_path = os.path.join(os.path.dirname(parentdir), run_name, model_name)
    task_args = {}
    with open(os.path.join(model_path, "args.yaml"), "r") as f:
        task_args_Loader = yaml.load_all(f, Loader = yaml.FullLoader)
        for t in task_args_Loader:
            task_args.update(t)
    args['state_mean'], args['state_std']= np.load(os.path.join(model_path, "model.state_mean.npy")), np.load(os.path.join(model_path, "model.state_std.npy"))
    body_mean, body_std = None, None
    
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args	["body_dim"]

    context_len = task_args["context_len"]      # K in decision transformer
    n_blocks = task_args['n_blocks']            # num of transformer blocks
    embed_dim = task_args['embed_dim']          # embedding (hidden) dim of transformer 
    n_heads = task_args['n_heads']              # num of transformer heads
    dropout_p = task_args['dropout_p']          # dropout probability
    device = torch.device(env_args.sim_device)
    pred_body = task_args.get('pred_body', True)
    EAT_model = LeggedTransformerBody(
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            state_mean=args['state_mean'], 
            state_std=args['state_std'],
            ).to(device)
    EAT_model.load_state_dict(torch.load(
        os.path.join(model_path,"model_best.pt")
    , map_location=device))
    EAT_model.eval()
    
    #testing
    file_path = os.path.join(os.path.dirname(parentdir), "evals", run_name, model_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = "AMP_EAT_Given_body_Extreme.csv"
    EAT_rows = np.arange(0.0, 1.0, 0.1)
    EAT_table = np.zeros((66,10))
    pair_list = []
    for i in range(12):#12条断腿情况
        for j in range(i+1,12):
            pair_list.append(str(i)+","+str(j))
    pair_list = np.array(pair_list)
    row_index = 0
    for i in range(12):#12条断腿情况
        for j in range(i+1,12):
        # for j in range (0, 0.8, 0.1):
            for column_index, k in enumerate(EAT_rows):            
                EAT_table[row_index, column_index], _ = test_EAT(args, env, EAT_model, [i,j], k, pred_body)
                # EAT_table[:,-1],_ = test_EAT(args, env, EAT_model, -1, 1) #测完好情况
                EAT_df = pd.DataFrame(EAT_table)
                EAT_df.index = pair_list
                EAT_df.columns = np.arange(0.0, 1.0, 0.1)
                EAT_res = EAT_df.to_csv(os.path.join(file_path, file_name), mode='w')
            row_index += 1
    