import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, Logger
from legged_gym.utils.task_registry_embody import task_registry
import numpy as np
import torch
from collections import deque
import statistics
import pandas as pd


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

def play(joint, rate, args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1000)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.5]# 更改速度设置以防命令采样到0的情况    

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # train_cfg.runner.load_run = "strange"
    train_cfg.runner.load_run = "Feb17_09-21-27_"
    train_cfg.runner.checkpoint = 1111
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
    import pdb 
    bodydim=12
    # rate = 0.9
    # t = torch.rand(env_cfg.env.num_envs)
    # p = torch.randint(1, bodydim+1, (env_cfg.env.num_envs,))
    # t = (t<rate) * p
    bodys = torch.ones(env_cfg.env.num_envs, bodydim)
    # print(bodys.shape)

    bodys[:,joint] = rate
    # print(bodys.shape)
    # print(bodys[0], bodys[1])
    # pdb.set_trace()
    # import random
    # for i in range(env_cfg.env.num_envs):
    #     if t[i] > 0:
    #         bodys[i][t[i]-1] = random.random()
    bodys = bodys.to(env.device)


    for i in range(2*int(env.max_episode_length)):
        # pdb.set_trace()
        # print(obs.shape)
        actions = policy(obs.detach(), bodys)
        # actions = disable_leg(actions, target="none", index=2)#let one joint or leg be disabled
        # obs, _, rews, dones, infos = env.step(actions.detach(), flawed_joint = [2], flawed_rate = 0.75)
        obs, _, rews, dones, infos = env.step(actions.detach(), bodys)

        cur_reward_sum += rews
        new_ids = (dones > 0).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length += torch.ones(env_cfg.env.num_envs,dtype=torch.float, device=args.sim_device)
        lengthbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        cur_episode_length[new_ids] = 0
        
        # if RECORD_FRAMES:
        #     if i % 2:
        #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
        #         env.gym.write_viewer_image_to_file(env.viewer, filename) 
        #         img_idx += 1 
        # if MOVE_CAMERA: #TODO: 这里可以设定视角变换，后续学习一下
        #     camera_position += camera_vel * env.dt
        #     env.set_camera(camera_position, camera_position + camera_direction)

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
            # pdb.set_trace()

    print(f"average reward is :{statistics.mean(rewbuffer)}\naverage length is :{statistics.mean(lengthbuffer)}\n")
    return statistics.mean(rewbuffer), statistics.mean(lengthbuffer)
            

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.experiment_name, 'export_3_5')
    if not os.path.exists(path):
        os.mkdir(path)
        Reward = np.zeros((12,10))
        Length = np.zeros((12,10))
        Re_df = pd.DataFrame(Reward, index = np.arange(12), columns = np.arange(0, 10)/10)
        Le_df = pd.DataFrame(Length, index = np.arange(12), columns = np.arange(0, 10)/10)
        Re_df.to_csv(os.path.join(path, "reward.csv"))
        Le_df.to_csv(os.path.join(path, "length.csv"))

    Re_df = pd.read_csv(os.path.join(path, "reward.csv"), index_col=0)
    Le_df = pd.read_csv(os.path.join(path, "length.csv"), index_col=0)
    reward, length = play(args.joint, args.rate_idx/10, args)
    Re_df.iat[args.joint, args.rate_idx] = reward
    Le_df.iat[args.joint, args.rate_idx] = length

    # for joint in range(12):  
    #     for rate_idx  in np.arange(0, 10):
    #         rate = rate_idx/10
    #         reward, length = play(joint, rate, args)
    #         Reward[joint][rate_idx] = reward
    #         Length[joint][rate_idx] = length
    # play(args)
    # Re_df = pd.DataFrame(Reward, index = np.arange(12), columns = np.arange(0, 10)/10)
    # Le_df = pd.DataFrame(Length, index = np.arange(12), columns = np.arange(0, 10)/10)

    

    Re_df.to_csv(os.path.join(path, "reward.csv"))
    Le_df.to_csv(os.path.join(path, "length.csv"))