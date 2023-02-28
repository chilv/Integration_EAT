import numpy as np
import os
from datetime import datetime

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args#, task_registry
import torch

import pickle

def collect(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    # env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_init_terrain_level = 9
    # env_cfg.commands.ranges.lin_vel_x = [0.4, 0.4]

    # privileged_dim = 203

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    #? 以下是否置为true以保证获得相对具有鲁棒性的轨迹？
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]# 更改速度设置以防命令采样到0的情况
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # ppo_runner, train_cfg = task_registry.make_teacher_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'strange'
    train_cfg.runner.checkpoint = 40000
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    device = ppo_runner.device
    _, _ = env.reset()
    env.episode_length_buf -= 1
    obs = env.get_observations()
    # privileged_obs = env.get_privileged_observations()
    # critic_obs = privileged_obs if privileged_obs is not None else obs
    obs = obs.to(device)
    policy = ppo_runner.get_inference_policy(device=ppo_runner.device)

    obs_buffer = torch.zeros(
        size=(env.num_envs, int(env.max_episode_length) + 1, env.num_obs), device=device)
    act_buffer = torch.zeros(size=(env.num_envs, int(env.max_episode_length) + 1, env.num_actions),
                                  device=device)

    total_steps = 0
    trajectories = []
    episode_length_before_reset = torch.zeros(env.num_envs,device=device, dtype=torch.long)

    for it in range(10000):
        # Rollout teacher policy to collect data
        with torch.inference_mode():
            episode_length_before_reset[:] = env.episode_length_buf
            clip_actions = env.cfg.normalization.clip_actions
            actions = policy(obs.detach())
            actions = torch.clip(actions, -clip_actions, clip_actions).to(device)
            obs_buffer[range(env.num_envs), episode_length_before_reset, :] = obs
            act_buffer[range(env.num_envs), episode_length_before_reset, :] = actions


            obs, privileged_obs, rewards, dones, infos = env.step(actions=actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, dones = obs.to(device), critic_obs.to(device), rewards.to(
                device), dones.to(device)

        env_ids = dones.nonzero(as_tuple=False).flatten()

        episode_length_before_reset += 1
        for i in range(len(env_ids)):
            env_id = env_ids[i]
            episode_data = {}
            episode_data['observations'] = obs_buffer[env_id,:episode_length_before_reset[env_id],:].cpu().numpy()
            episode_data['actions'] = act_buffer[env_id, :episode_length_before_reset[env_id], :].cpu().numpy()
            total_steps += (episode_length_before_reset[env_id]).cpu().numpy()
            trajectories.append(episode_data)

            print('total steps:', total_steps)

        if(total_steps > 20000000):
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'data', 'data.pkl')                
            with open(filename, 'wb') as f:
                pickle.dump(trajectories, f)
            break



if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    collect(args)
