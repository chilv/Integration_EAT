
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from scripts.utils import flaw_generation
from model import LeggedTransformerPro

import numpy as np
import pandas as pd
import torch
from collections import deque
import statistics

codename_list = []	#存储每条腿的字母代号
for i in ["F", "B"]:
    for j in ["L", "R"]:
        for k in ["H", "K", "A"]:
            codename_list.append(j+i+k)

ENV_NUMS = 1024  #测试环境数

def test_ppo(args, env, train_cfg, faulty_tag = -1, flawed_rate = 1):
    """在单次循环中

    Args:
        args (_type_): 就各种参数
        env ( optional): 用来测试的环境
        train_cfg ( optional): 用来训练脚本
        faulty_tag (int, optional): 坏的关节 -1为全好. Defaults to -1.
        flawed_rate (int, optional): 坏的成都  1为完好. Defaults to 1.
    """
    
    obs = env.reset()[0]
    # obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # train_cfg.runner.load_run = "strange"
    if faulty_tag == -1:
        train_cfg.runner.load_run = f""
    else:
        train_cfg.runner.load_run = f"{flawed_rate}_torques/{codename_list[faulty_tag]}_{flawed_rate}"
        
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    #判断模型文件是否存在 若不存在则报错弹出
    if not os.path.exists(os.path.join(log_root,train_cfg.runner.load_run)):
        print(f"no model file{faulty_tag}_{flawed_rate}")
        return -1 , 0

    train_cfg.runner.checkpoint = -1    #! 改成best
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    cur_reward_sum = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    rewbuffer = deque(maxlen=100)
    cur_episode_length = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    lengthbuffer = deque(maxlen=100)
    
    for i in range(int(env.max_episode_length) + 1):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach(), flawed_joint = [faulty_tag], flawed_rate = flawed_rate)
        
        cur_reward_sum += rews
        new_ids = (dones > 0).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length += torch.ones(env_cfg.env.num_envs,dtype=torch.float, device=args.sim_device)
        lengthbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        cur_episode_length[new_ids] = 0

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i % stop_rew_log == 0 and i != 0:
            logger.print_rewards()
            if len(rewbuffer)>0:
                print(f"average reward is :{statistics.mean(rewbuffer)}\naverage length is :{statistics.mean(lengthbuffer)}\n")
    return statistics.mean(rewbuffer), statistics.mean(lengthbuffer)
            
def test_EAT(args, env, EAT_model, pass_args, faulty_tag = -1, flawed_rate = 1):
    # loading EAT model
    eval_batch_size = ENV_NUMS        # 测试环境数
    max_test_ep_len = 1000    	#iters

    body_target = [1 for _ in range(12)]
    if (faulty_tag != "none"):
        body_target[faulty_tag] = flawed_rate
    body_dim = len(body_target)
    
    state_mean = torch.from_numpy(pass_args["state_mean"]).to(device)
    state_std = torch.from_numpy(pass_args["state_std"]).to(device)
    # body_mean = torch.from_numpy(pass_args["body_mean"]).to(device)
    # body_std = torch.from_numpy(pass_args["body_std"]).to(device)
    
    body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
    bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32)
    
    results = {}
    #testing loop
    running_state = env.reset()[0]
    running_reward = torch.zeros((eval_batch_size, ), dtype=torch.float32, device=device)
    
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
    actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
    states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                        dtype=torch.float32, device=device)
    total_rewards = np.zeros(eval_batch_size)
    total_length = np.zeros(eval_batch_size)
    dones = np.zeros(eval_batch_size)
    
    #testing 
    with torch.no_grad():
        print(f"joint {codename_list[faulty_tag]} with flawed rate {flawed_rate} is under testing")
        for t in range(max_test_ep_len):

            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            if t < context_len:
                _, act_preds, _ = EAT_model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            body=bodies[:,:context_len])
                act = act_preds[:, t].detach()
            else:
                _, act_preds, _ = EAT_model.forward(timesteps[:,t-context_len+1:t+1],
                                        states[:,t-context_len+1:t+1],
                                        actions[:,t-context_len+1:t+1],
                                        body=bodies[:,t-context_len+1:t+1])
                act = act_preds[:, -1].detach()
            
            # body, _ = flaw_generation(ENV_NUMS, fixed_joint = [faulty_tag], flawed_rate = flawed_rate, device = )
            running_state, _, running_reward, done, infos = env.step(act, [faulty_tag], flawed_rate)
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

if __name__ == '__main__':
    args = get_args()
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUMS)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]# 更改速度设置以防命令采样到0的情况    

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    #测试ppo======================================================================
    # ppo_row_names = [0, 0.25, 0.5, 0.75]
    # out_table = np.zeros((12,5))
    # for i in range(12):#12条断腿情况
    #     # for j in range (0, 0.8, 0.1):
    #     for j in ppo_row_names:            
    #         out_table[i, ppo_row_names.index(j)], _ = test_ppo(args, env, train_cfg, i, j)
    # out_table[:,-1],_ = test_ppo(args, env, train_cfg, -1, 1) #测完好情况
    # ppo_df = pd.DataFrame(out_table)
    # ppo_df.index = codename_list
    # ppo_df.columns = [0,0.25,0.5, 0.75, 1.0]
    # ppo_res = ppo_df.to_csv(os.path.join(LEGGED_GYM_ROOT_DIR,"logs/fualty_sppo_.csv"), mode='w')
    # np.savetxt(os.path.join(LEGGED_GYM_ROOT_DIR,"logs/fualty_ppo.csv"), out_table, delimiter=',')
    #测试ppo结束===================================================================
    
    
    #测试EAT======================================================================
    #loading EAT model
    # loading pre_record stds,means...
    model_path = os.path.join(parentdir, "EAT_runs/EAT_FLAWEDPPO_09/")
    state_mean, state_std, body_mean, body_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy"), None, None#, np.load(model_path+"model.body_mean.npy"), np.load(model_path+"model.body_std.npy")
    pass_args = {"state_mean":state_mean, "state_std":state_std, "body_mean":body_mean, "body_std":body_std}
    
    state_dim = 48
    act_dim = 12
    body_dim = 12	

    context_len = 50      # K in decision transformer
    n_blocks = 6            # num of transformer blocks
    embed_dim = 256          # embedding (hidden) dim of transformer 
    n_heads = 1              # num of transformer heads
    dropout_p = 0.1          # dropout probability
    device = torch.device(args.sim_device)
    EAT_model = LeggedTransformerPro(
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            state_mean=state_mean,
            state_std=state_std,
            body_mean=body_mean,
            body_std=body_std
            ).to(device)
    EAT_model.load_state_dict(torch.load(
        os.path.join(model_path,"model_best.pt")
    ))
    EAT_model.eval()
    
    #testing
    EAT_rows = np.arange(0.0, 1.0, 0.1)
    EAT_table = np.zeros((12,10))
    for i in range(12):#12条断腿情况
        # for j in range (0, 0.8, 0.1):
        for j in EAT_rows:            
            EAT_table[i, np.where(EAT_rows==j)], _ = test_EAT(args, env, EAT_model, pass_args, i, j)
    EAT_table[:,-1],_ = test_EAT(args, env, EAT_model, pass_args, -1, 1) #测完好情况
    EAT_df = pd.DataFrame(EAT_table)
    EAT_df.index = codename_list
    EAT_df.columns = np.arange(0.0, 1.0, 0.1)
    EAT_res = EAT_df.to_csv(os.path.join(LEGGED_GYM_ROOT_DIR,"logs/tables/fualty_EAT_09_small.csv"), mode='w')
    # np.savetxt(os.path.join(LEGGED_GYM_ROOT_DIR,"logs/fualty_EAT2.csv"), EAT_table, delimiter=',')
    #测试EAT结束===================================================================
    