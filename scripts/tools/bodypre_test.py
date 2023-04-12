"""
此脚本用于测试对于body的预测是否准确
"""
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
# from legged_gym import LEGGED_GYM_ROOT_DIR

import yaml
import numpy as np
import pandas as pd

import isaacgym
import torch
from argparse import Namespace

from legged_gym.envs import *
from legged_gym.utils import task_registry, Logger, get_args
from model import LeggedTransformerBody
import pdb

def bodypre_test(argpass, model, env, faulty_tag:int = -1, flawed_rate:int = 1):
    #===================================================================
    # arg init
    max_test_ep_len=1000    #iters
    eval_batch_size = argpass["eval_batch_size"]  # envs
    state_dim = argpass["state_dim"]
    act_dim = argpass["act_dim"]
    body_dim = argpass["body_dim"]
    device = argpass["device"]
    context_len = argpass["context_len"]
    
    #====================================================================================    
    # zeros place holders
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
    states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                        dtype=torch.float32, device=device)
    actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                        dtype=torch.float32, device=device)
    body_oringe = [1 for _ in range(12)]
    body_dim = len(body_oringe)
    # init episode
    running_state = env.reset()[0]
    running_reward = torch.zeros((eval_batch_size, ),
                        dtype=torch.float32, device=device)

    #log init
    logger = Logger(env.dt)
    results = {}
    total_timesteps = 0
    total_rewards = np.zeros(eval_batch_size)
    dones = np.zeros(eval_batch_size)
    
    #=======================================================
    # evaluating
    with torch.no_grad():
        wrong_pre = 0
        pre_diff = 0
        for t in range(max_test_ep_len):
            total_timesteps += (dones == 0)
            
            faulty_taget = faulty_tag
            body_target = body_oringe.copy()
            body_target[faulty_taget] = flawed_rate
            body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
            bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()

            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            if t < context_len:
                _, act_preds, body_preds = model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            body=bodies[:,:context_len])
                act = act_preds[:, t].detach()
                # body = body_preds[:, t].detach()
            else:
                _, _, body_preds = model.forward(timesteps[:,t-context_len+1:t+1],
                                        states[:,t-context_len+1:t+1],
                                        actions[:,t-context_len+1:t+1],
                                        body=bodies[:,t-context_len+1:t+1])
                bodies[:, t] = body_preds[:, -1].detach()   #加这一句可以让学出来的body返回回去
                for i in range(eval_batch_size):
                    if torch.argmin(bodies[i,t,:]) != faulty_taget:
                        wrong_pre += 1

                pre_diff += bodies[:,t,faulty_tag] - bodies[:,0,faulty_tag]
                # if t % 900 == 0 and t > 0:
                #     pdb.set_trace()
                # print(torch.argmin(bodies[:,t,:], dim=-1))
                _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                        states[:,t-context_len+1:t+1],
                                        actions[:,t-context_len+1:t+1],
                                        body=bodies[:,t-context_len+1:t+1])
                                        # body=bodies[:,t-context_len+1:t+1] if not nobody else None)
                act = act_preds[:, -1].detach()
            
            running_state, _, running_reward, done, infos = env.step(act, flawed_joint = [faulty_taget], flawed_rate = flawed_rate) #if t > max_test_ep_len/8 else env.step(act, [-1])
            
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
                
            # if RECORD_FRAMES:
            #     if t % 2:
            #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            #         env.gym.write_viewer_image_to_file(env.viewer, filename) 
            #         img_idx += 1 
            # if MOVE_CAMERA: #TODO: 这里可以设定视角变换，后续学习一下
            #     camera_position += camera_vel * env.dt
            #     env.set_camera(camera_position, camera_position + camera_direction)
            
            actions[:, t] = act

            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            dones += done.detach().cpu().numpy()

            if torch.all(done):
                break

    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_timesteps)
    results['eval/right_rate'] = 1 - ( wrong_pre / ((max_test_ep_len-20) * eval_batch_size) )
    # pdb.set_trace()
    results['eval/aver_diff'] = np.mean(pre_diff.detach().cpu().numpy())/(max_test_ep_len-20)
    
    print(f"tag{faulty_taget} rate{flawed_rate}'s \
        average reward is :{results['eval/avg_reward']} \
          \naverage length is :{results['eval/avg_ep_len']} \
          \nwrong pre rate is {results['eval/right_rate']} \
        \naverage diff is {results['eval/aver_diff']} \
        \n")
    logger.print_rewards()
    
    return results['eval/avg_reward'], results['eval/right_rate'], results['eval/aver_diff']  #reward, unprecious_rate, aver_diff
    

    
if __name__ == "__main__":
    with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)
    
    #==========================================
    #some args init
    device = torch.device("cuda:0")         #setting flexible
    
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args["body_dim"]

    context_len = args["context_len"]      # K in decision transformer
    n_blocks = args["n_blocks"]            # num of transformer blocks
    embed_dim = args["embed_dim"]          # embedding (hidden) dim of transformer
    n_heads = args["n_heads"]              # num of transformer heads
    dropout_p = args["dropout_p"]          # dropout probability
    model_name = "EAT++20mse_IPPO3_02/"
    
    eval_batch_size = 100  # envs
    
    #===============================================================
    #prepare model
    print("loading pre_record stds,means...")
    model_path = os.path.join("./Integration_EAT/EAT_runs/", model_name)
    
    state_mean, state_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy")
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)
    
    model = LeggedTransformerBody(
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p
            ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_path,"model4000epoch.pt"), map_location = "cuda:0"
    ))
    model.eval()
    
    #======================================================================
    #prepare envs
    env_args = get_args()
    
    env_cfg, _ = task_registry.get_cfgs(name=args["task"])
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, eval_batch_size)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]# 更改速度设置以防命令采样到0的情况    

    # args = Namespace(**args)
    env, _ = task_registry.make_env(name=args["task"], args=env_args, env_cfg=env_cfg)
    
    #=========================================================================
    # eval loop
    argpass = {
        "state_mean": state_mean, 
        "state_std": state_std,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "body_dim": body_dim,
        "context_len": context_len,
        "device": device,
        "eval_batch_size": eval_batch_size
    }
    tags = range(12)
    rates = np.arange(0, 0.5, 0.05)
    Rews = np.zeros((len(tags), len(rates)))
    Prec = np.zeros((len(tags), len(rates)))
    Diff = np.zeros((len(tags), len(rates)))
    for tag in tags:
        for rate in rates:
            # pdb.set_trace()
            Rews[tag, int(rate*20)],Prec[tag, int(rate*20)],Diff[tag, int(rate*20)] = bodypre_test(argpass, model, env, faulty_tag = tag, flawed_rate = rate)
    
    EAT_df = pd.DataFrame(np.concatenate(Rews, Prec, Diff), axis = 1)
    EAT_df.index = tags
    EAT_df.columns = rates*3
    EAT_res = EAT_df.to_csv(os.path.join("./Integration_EAT/scripts/tools/bodypre_analyse", f"{model_name}.csv"), mode='w')
    print("file written")
    
    