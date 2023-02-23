import pdb
# import pickle
# from legged_gym import LEGGED_GYM_ROOT_DIR
import os

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import gym


class A1(gym.Env):
    def __init__(self, num_envs=None, robot="a1", discrete=None, policy_robot=None, noise=0, device=None):
        super(A1, self).__init__()

        self.observation_shape = (48,)
    
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)

        self.num_envs = num_envs

        if policy_robot is None:
            policy_robot = robot
        env_cfg, _ = task_registry.get_cfgs(name=robot)
        _, train_cfg = task_registry.get_cfgs(name=policy_robot)
        # override some parameters for testing
        env_cfg.env.num_envs = 1 if self.num_envs is None else self.num_envs
        # pdb.set_trace()

        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        # if flat:
        #     # pass
        #     env_cfg.terrain.mesh_type = "plane"
        env_cfg.noise.add_noise = bool(noise)
        env_cfg.noise.noise_level = noise
        env_cfg.domain_rand.randomize_friction = bool(noise)
        env_cfg.domain_rand.push_robots = bool(noise)

        env_cfg.commands.ranges.lin_vel_x = [0.3,0.7]
        env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

        # discrete terrain
        if discrete is not None:
            env_cfg.terrain.mesh_type = ["heightfield", "trimesh"][1]
            env_cfg.terrain.selected = True
            env_cfg.terrain.terrain_kwargs = {"difficulty": float(discrete)}
            env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1]

        # prepare environment
        self.env, _ = task_registry.make_env(name=robot, args=None, env_cfg=env_cfg)
        self._max_episode_steps = 1000
        # pdb.set_trace()
        # obs = env.get_observations()

        # train_cfg.runner.resume = True
        ppo_runner, _ = task_registry.make_alg_runner(env=self.env, name=policy_robot, args=None, train_cfg=train_cfg)
        self.ppo_policy = ppo_runner.get_inference_policy(device=device)

    def reset(self):
        obs, _ = self.env.reset()
        # pdb.set_trace()
        if self.num_envs is None:
            return np.squeeze(obs.detach().cpu().numpy())
        else:
            return obs
        
    def step(self, action, flawed_joint = [-1], flawed_rate = 1):

        # if broken:
        #     # action[2] = 0 #RF
        #     # action[5] = 0 #LF
        #     # action[1] = 1 #RF
        #     # action[2] = -1 #LF

        #     # action[1] = 0
        #     # action[2] = 0
        #     # action[4] = 0
        #     # action[5] = 0

        #     # action[1] = 0
        #     # action[2] = 0
        #     # action[4] = 0
        #     # action[5] = 0
        #     action[6] = 0
        #     action[7] = 0
        #     action[8] = 0
        #     action[9] = 0
        #     action[10] = 0
        #     action[11] = 0


        if self.num_envs is None:
            actions = torch.Tensor(np.array([action]))
            obs, _, rews, dones, infos = self.env.step(actions.detach(), flawed_joint, flawed_rate)
            s = np.squeeze(obs.detach().cpu().numpy())
            # print("COMMAND: ", s[9:12])
            # print("POS: ", self.env.root_states[0, :2])

            lookat = self.env.root_states[0, :3]  # [m]
            camera_position = lookat.detach().cpu().numpy() + [2,2,2]  # [m]
            self.env.set_camera(camera_position, lookat)

            return s, rews.detach().cpu().numpy().item(), dones.detach().cpu().numpy().item(), infos

        else:
            obs, _, rews, dones, infos = self.env.step(action.detach(), flawed_joint, flawed_rate)
            return obs.detach(), rews.detach(), dones.detach(), infos

    def get_normalized_score(self, score):
        return score

    def close(self):
        self.env.close()


        # pdb.set_trace()
        # self.env.gym.write_viewer_image_to_file(self.env.viewer,"test.png")
        # self.env.gym.get_camera_view_matrix(self.env.sim, self.env.gym.get_viewer_camera_handle(self.env.viewer))
        # pdb.set_trace()

        



        # for i in range(int(env.max_episode_length-1)):
        #     # actions = policy(obs.detach())
        #     actions = torch.Tensor([[0]*12])

        #     obs, _, rews, dones, infos = env.step(actions.detach())

# a1 = A1()
# a1.reset()

# for _ in range(1000):
#     a1.step(a1.action_space.sample())