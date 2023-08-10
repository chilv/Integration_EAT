# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

#trying faulty leg

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from collections import deque
import statistics


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
        actions[:,index]= -1.0 #将指定索引的关节置0，暂定左前腿的2号关节失能
    elif target == "leg" :
        actions[:, 3*index:3*index+3] = -1.0
    else:
        pass
        
    return actions

def play(args, flawed_joint = -1, flawed_rate = 1):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0.7]# 更改速度设置以防命令采样到0的情况    
    env_cfg.commands.ranges.ang_vel_yaw = [0.5, 0.5]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # train_cfg.runner.load_run = "strange"
    train_cfg.runner.load_run = "PPO_Models"
    train_cfg.runner.checkpoint = flawed_joint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    cur_reward_sum = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    rewbuffer = deque(maxlen=100)
    cur_episode_length = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    lengthbuffer = deque(maxlen=100)
    bodydim = 12
    
    bodies = torch.ones(env_cfg.env.num_envs, bodydim)
    bodies[:, flawed_joint] = flawed_rate     #后续泛化
    bodies = bodies.to(env.device)
    
    for i in range(2*int(env.max_episode_length)):
        actions = policy(obs.detach(),bodies)
        # actions = disable_leg(actions, target="none", index=2)#let one joint or leg be disabled
        obs, _, rews, dones, infos = env.step(actions.detach(), bodies)
        
        cur_reward_sum += rews
        new_ids = (dones > 0).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length += torch.ones(env_cfg.env.num_envs,dtype=torch.float, device=args.sim_device)
        lengthbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        cur_episode_length[new_ids] = 0
        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename) 
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[9, :3]
            # camara_position = lootat.detach().cpu().numpy() + [0, 1, 0.5]
            camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
            # camara_position = lootat.detach().cpu().numpy() + [-1, 0, 0]
            env.set_camera(camara_position, lootat)
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # elif i==stop_state_log:
            # logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i % stop_rew_log == 0:
            logger.print_rewards()
            if len(rewbuffer)>0:
                print(f"average reward is :{statistics.mean(rewbuffer)}\naverage length is :{statistics.mean(lengthbuffer)}\n")
                      
            

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args, 4, 1)