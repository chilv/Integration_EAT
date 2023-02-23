import argparse
from argparse import Namespace
import os
import pickle
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from datetime import datetime

import numpy as np

from legged_gym.envs import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from utils import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score, evaluate_on_env_batch_body, get_dataset_config
from model import DecisionTransformer, LeggedTransformer, LeggedTransformerPro, MLPBCModel
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from singlea1 import A1
import statistics

from tqdm import trange, tqdm


def disable_leg(actions, target:str ="joint", index:int = 2):
    """用以让机器狗某条退或者某个关节失能，
    暂时只提供单关节失能用以初步测试

    Args:
        actions (_type_): 原动作
        target: 标识需要失能的目标类别
        index: 需要失能的具体目标

    Returns:
        _type_: 失能后动作
    """
    if target == "joint" :
        actions[:,index]=0 #将指定索引的关节置0，暂定左前腿的2号关节失能
    elif target == "leg" :
        actions[:, 3*index:3*index+3] = -1.0
    else:
        pass
        
    return actions

def play(args, faulty_tag = -1):
    rtg_scale = 1000      # normalize returns to go
    state_dim = 48
    act_dim = 12
    body_dim = 12

    context_len = 20      # K in decision transformer
    n_blocks = 6            # num of transformer blocks
    embed_dim = 128          # embedding (hidden) dim of transformer #! 原值128 #512
    n_heads = 1              # num of transformer heads
    dropout_p = 0.1          # dropout probability
    
    #======================================================================
    #prepare datas(for init model)  
    # datafile, i_magic_list, eval_body_vec, eval_env = get_dataset_config("top1000")
    
    # file_list = [f"{i_magic}-{datafile}.pkl" for i_magic in i_magic_list]
    # dataset_path_list_raw = [os.path.join('EAT-main/data/', d) for d in file_list]
    # dataset_path_list = []
    # for p in dataset_path_list_raw:
    #     if os.path.isfile(p):
    #         dataset_path_list.append(p)
    #     else:
    #         print(p, " doesn't exist~")
                
    # device = torch.device(args.sim_device)
    
    # print("Loding paths for each robot model...")
    # #增加一部分用于记录轨迹各项均值/标准差的
    # big_list = []
    # state_recorde = {}    
    # for pkl in tqdm(dataset_path_list):
    #     tmp_state_mean, tmp_state_std, tmp_body_mean, tmp_body_std = 0,0,0,0
    #     with open(pkl, 'rb') as f:
    #         thelist = pickle.load(f)
    #         traj_states,traj_bodies = [],[]
    #         for traj in thelist:
    #             traj_states.append(traj['observations'])
    #             traj_bodies.append(traj['body'])
    #         # used for input normalization
    #         traj_states = np.concatenate(traj_states, axis=0)
    #         traj_bodies = np.concatenate(traj_bodies, axis=0)
    #         tmp_state_mean, tmp_state_std = np.mean(traj_states, axis=0), np.std(traj_states, axis=0) + 1e-6
    #         tmp_body_mean, tmp_body_std = np.mean(traj_bodies, axis=0), np.std(traj_bodies, axis=0) + 1e-6

    #         state_recorde[pkl.split('/')[-1][:-4]] = {"state_mean":tmp_state_mean, "state_std":tmp_state_std, "body_mean":tmp_body_mean, "body_std":tmp_body_std}
    #     assert "body" in thelist[0]
    #     big_list = big_list + thelist
    # traj_dataset = D4RLTrajectoryDataset(big_list, context_len, rtg_scale, leg_trans_pro=True)
    # assert body_dim == traj_dataset.body_dim
    
    # state_mean, state_std, body_mean, body_std = traj_dataset.get_state_stats(body=True)
    # state_recorde["total"] = {"state_mean":state_mean, "state_std":state_std, "body_mean":body_mean, "body_std":body_std}
    # with open("EAT-main/data/top1000_recorde.pkl", 'wb') as f:
    #     pickle.dump(state_recorde, f)
    #=========================================================================================================
    print("loading pre_record stds,means...")
    # with open("EAT-main/data/state_recorde.pkl", 'rb') as f:
    #     dict_record = pickle.load(f)
    #     # 以下使用独立均值：
    #     # if(faulty_tag>=5):#由于没有左前右前的踝关节故障模型 所以坏损关节标号和轨迹标号需要重新对偶
    #     #     the_tag = faulty_tag - 2
    #     # elif(faulty_tag >= 2):
    #     #     the_tag = faulty_tag - 1
    #     # else:
    #     #     the_tag = faulty_tag
    #     # the_record = dict_record[file_list[the_tag + 1][:-4]]    #faulty_tag + 1是由于 -1代表着完好的机器狗，和file_list的储存方式错开一位
    #     # 以下使用全局均值(optional)：
    #     the_record = dict_record["total"]
        
    #     state_mean, state_std, body_mean, body_std = the_record["state_mean"],the_record["state_std"],the_record["body_mean"],the_record["body_std"]
    save_model_path = "/NAS2020/Workspaces/DRLGroup/wentaodong/Integration_EAT/EAT_runs/EAT_FID2341_01/model"
    state_mean = np.load(f"{save_model_path}.state_mean.npy")
    state_std = np.load(f"{save_model_path}.state_std.npy")
    body_mean = np.load(f"{save_model_path}.body_mean.npy")
    body_std = np.load(f"{save_model_path}.body_std.npy")


    #======================================================================
    #prepare envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 25)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.5]# 更改速度设置以防命令采样到0的情况    

    # env = A1(num_envs=args.num_eval_ep, noise=args.noise)     #另一种环境初始化方式
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    #===================================================================================
    
    
    #====================================================================================
    # prepare algs
    device = torch.device(args.sim_device)
    model = LeggedTransformerPro(
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
    model.load_state_dict(torch.load(
        "/NAS2020/Workspaces/DRLGroup/wentaodong/Integration_EAT/EAT_runs/EAT_FID2341_01/model_best.pt"
    ))
    model.eval()
    #====================================================================================
    
        
    #====================================================================================
    ##eval pre
    #init visualize
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    #init eval para
    eval_batch_size = 25  # envs
    max_test_ep_len=1000    #iters
    nobody = False
    body_target = [0 for _ in range(12)]
    if (faulty_tag != -1):
        body_target[faulty_tag] = 1
    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = 48
    act_dim = 12
    body_dim = len(body_target)
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)
    body_mean = torch.from_numpy(body_mean).to(device)
    body_std = torch.from_numpy(body_std).to(device)
    
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    with torch.no_grad():
        if True:
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            body_target = (torch.tensor(body_target, dtype=torch.float32, device=device) - body_mean) / body_std
            bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32)

            # init episode
            running_state = env.reset()[0]
            running_reward = torch.zeros((eval_batch_size, ),
                                dtype=torch.float32, device=device)

            total_rewards = np.zeros(eval_batch_size)
            dones = np.zeros(eval_batch_size)

            for t in range(max_test_ep_len):

                total_timesteps += 1

                states[:,t,:] = running_state
                states[:,t,:] = (states[:,t,:] - state_mean) / state_std

                if t < context_len:
                    if not nobody:
                        _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                    states[:,:context_len],
                                                    actions[:,:context_len],
                                                    body=bodies[:,:context_len])
                    else:
                        _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                    states[:,:context_len],
                                                    actions[:,:context_len])
                    act = act_preds[:, t].detach()
                else:
                    if not nobody:
                        _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                body=bodies[:,t-context_len+1:t+1] if not nobody else None)
                    else:
                        _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1])
                    act = act_preds[:, -1].detach()

                #let one joint or leg be disabled
                act = disable_leg(act.detach(), target="none", index=3)
                # running_state, running_reward, done, _ = env.step(act.cpu())
                
                running_state, _, running_reward, done, infos = env.step(act, [faulty_tag])
                # if t < max_test_ep_len/8:
                    # running_state, _, running_reward, done, infos = env.step(act, [-1])
                
                
                # if RECORD_FRAMES:
                #     if t % 2:
                #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                #         env.gym.write_viewer_image_to_file(env.viewer, filename) 
                #         img_idx += 1 
                # if MOVE_CAMERA: #TODO: 这里可以设定视角变换，后续学习一下
                #     camera_position += camera_vel * env.dt
                #     env.set_camera(camera_position, camera_position + camera_direction)
            
                actions[:, t] = act

                total_reward += np.sum(running_reward.detach().cpu().numpy())
                total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
                dones += done.detach().cpu().numpy()

                if torch.all(done):
                    break

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.sum(total_rewards) / eval_batch_size
    results['eval/avg_ep_len'] = total_timesteps
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='fid234-1')

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=1000)
    parser.add_argument('--noise', type=int, help="noisy environemnt for evaluation", default=0)

    parser.add_argument('--dataset_dir', type=str, default='Integraton_EAT/data/')
    parser.add_argument('--log_dir', type=str, default='Integraton_EAT/EAT_runs/')
    parser.add_argument('--cut', type=int, default=0)

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=6)
    # parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--n_epochs_ref', type=float, default=1)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--nobody', default=False, action='store_true', help="use DT")

    parser.add_argument('--wandboff', default=False, action='store_true', help="Disable wandb")

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    args = get_args()
    
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    play(args, 11)
