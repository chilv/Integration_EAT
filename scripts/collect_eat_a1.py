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

import pdb
import pickle
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))

from legged_gym import LEGGED_GYM_ROOT_DIR
import collections
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from tqdm import trange, tqdm

NUM_ENVS = 10000 #10000 # 400 #4000 #1000 # 50# 20000 #
REP = 1 #10 #20
ZERO_VEL = False
VEL_TO_ACC = False
NOISE = False
TOPK = 0 #0 #1000 # 10000
PASSSCORE = 20	#??????????????????
FILTER = True
VEL0_5 = False
VEL0_4 = False
SAVE_DIR = os.path.join(parentdir, "data")

# param_dict = {}
## ???????????????i_magic??????????????????12??????1???????????? ???????????????????????????
## 12??????????????????????????????????????? ???????????????????????????
## ??????????????????L?????? R?????? F?????? B?????? H?????? K?????? A?????? ????????????????????????????????????
codename_list = []	#??????????????????????????????
for i in ["F", "B"]:
    for j in ["L", "R"]:
        for k in ["H", "K", "A"]:
            codename_list.append(j+i+k)
rate_list = [0, 0.25, 0.5, 0.75]#?????????????????????????????????


def play(args, env, train_cfg, fault_type = "none", fault_rate = 1):
	'''
	??????????????????????????????????????????????????????????????????????????????????????????????????????
	----------------------------
	????????????:
	args: ??????????????????
	env: ?????????????????????
	train_cfg: ????????????
	fault_type: ???????????????????????????none??????????????????
	fault_rate: ?????????????????????1?????????????????????????????????????????????
	----------------------------
	'''
	print(f"collecting {fault_type} with flawed_rate {fault_rate} ...")
	env.reset()
	obs = env.get_observations()
	# load policy
	train_cfg.runner.resume = True
	train_cfg.runner.load_run = ""
	train_cfg.runner.checkpoint = -1
	model_root = os.path.join(parentdir, "models")
	if fault_type != "none":	#?????????????????????????????????????????????
		train_cfg.runner.load_run = fault_type + "_" + str(fault_rate)
		model_root = os.path.join(model_root, str(fault_rate) + "_torques")
		#?????????????????????????????? ???????????????????????????
		if not os.path.exists(os.path.join(model_root,train_cfg.runner.load_run)):
			print(f"no model file {fault_type}_{fault_rate}")
			return fault_type + "_" + str(fault_rate) + " file not exists"
	ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=model_root)
	policy = ppo_runner.get_inference_policy(device=env.device)

	data_set = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[], 'timeouts':[]}

	special = ""
	# if ZERO_VEL:
	# 	special += "-zerovel"
	# if VEL0_5:
	# 	special += "-vel0.5"
	# if VEL0_4:
	# 	special += "-vel0.4"
	# if NOISE:
	# 	special += "-n"
    		
	# OUTPUT_FILE = "DTdata/a1magic4-uncle-v2.pkl"
	ver_index = 0
	if not TOPK:
		output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	else:
		output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# if not PASSSCORE:
	# 	output_file = f"{SAVE_DIR}{fault_type}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# else:
	# 	output_file = f"{SAVE_DIR}{fault_type}-P{PASSSCORE}F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"

	while os.path.isfile(output_file):
		ver_index += 1
		if not TOPK:
			output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
		else:
			output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"

	print("Preparing file to ", output_file)

	total_rewards = np.zeros(NUM_ENVS)
	total_dones = np.zeros(NUM_ENVS)


	print("RECORDING DATA ......")
	assert int(env.max_episode_length) == 1000
	for i in trange(REP*int(env.max_episode_length)):
		actions = policy(obs.detach())
		# pdb.set_trace()
		if not ZERO_VEL:
			data_set['observations'].append(obs.cpu().detach().numpy()[:,:48])
		else:
			obs_ori = obs.cpu().detach().numpy()[:,:48]
			obs_ori[:,:3] = np.zeros(3)
			data_set['observations'].append(obs_ori)
		data_set['actions'].append(actions.cpu().detach().numpy())
		if fault_type == "none":
			obs, _, rews, dones, infos = env.step(actions.detach())
		else:
			obs, _, rews, dones, infos = env.step(actions.detach(), flawed_joint = [codename_list.index(fault_type)], flawed_rate = fault_rate)

		data_set['rewards'].append(rews.cpu().detach().numpy())
		data_set['terminals'].append(dones.cpu().detach().numpy())
		# data_set['timeouts'].append(np.array([i == int(env.max_episode_length-1)]*NUM_ENVS))
		if not ZERO_VEL:
			data_set['next_observations'].append(obs.cpu().detach().numpy()[:,:48])
		else:
			obs_ori = obs.cpu().detach().numpy()[:,:48]
			obs_ori[:,:3] = np.zeros(3)
			data_set['next_observations'].append(obs_ori)

		data_set['timeouts'].append(infos["time_outs"].cpu().detach().numpy())
		data_set['timeouts'][-1] = np.array([True]*NUM_ENVS) if i == int(env.max_episode_length-1) else data_set['timeouts'][-1]

		#? ?????????rew??????????????????????????????????????????????????????
		total_rewards += rews.detach().cpu().numpy() * (total_dones == 0)
		total_dones += dones.detach().cpu().numpy()


	print("MEAN SCORE: ", np.mean(total_rewards))

	print("[REORGANISING DATA ......]")

	keys = ["observations", "next_observations", "actions", "rewards", "terminals", "timeouts"]

	for k in keys:
		print("Preprocessing ", k)
		data_set[k] = np.array(data_set[k])


	obss = np.array(data_set['observations']).transpose((1,0,2))
	nobss = np.array(data_set['next_observations']).transpose((1,0,2))
	acts = np.array(data_set['actions']).transpose((1,0,2))
	ds = np.array(data_set['terminals']).transpose()
	rs = np.array(data_set['rewards']).transpose()

	paths = []
	for obs_p, nobs_p, act_p, rew_p, done_p in zip(obss, nobss, acts, rs, ds):
		obs_list = []
		nobs_list = []
		act_list = []
		rew_list = []
		done_list = []
		path_dict = {}

		for obs_t, nobs_t, act_t, rew_t, done_t in zip(obs_p, nobs_p, act_p, rew_p, done_p):
			obs_list.append(obs_t)
			nobs_list.append(nobs_t)
			act_list.append(act_t)
			rew_list.append(rew_t)
			done_list.append(done_t)
			if done_t:
				break

		path_dict['observations'] = np.array(obs_list)
		path_dict['next_observations'] = np.array(nobs_list)
		path_dict['actions'] = np.array(act_list)
		path_dict['rewards'] = np.array(rew_list)
		path_dict['terminals'] = np.array(done_list)

		embodiment = [1 for _ in range(12)]
		if fault_type != "none":
			embodiment[codename_list.index(fault_type)] = fault_rate #??????????????????embody???????????????
		path_dict["body"] = np.tile(np.array(embodiment), (np.shape(path_dict["observations"])[0], 1))

		paths.append(path_dict)

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	# pdb.set_trace()
	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return fault_type + "_" + str(fault_rate) + "no traj"

	if not PASSSCORE:
		top10000 = (-returns).argsort()[:TOPK]
	else:
		top10000 = np.nonzero(returns > PASSSCORE)[0]
	paths_out = []
	for i in tqdm(top10000):
		paths_out.append(paths[i])

	print("-->")

	returns = np.array([np.sum(p['rewards']) for p in paths_out])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths_out])
	print(f'Number of samples collected: {num_samples}')

	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return fault_type + "_" + str(fault_rate) + "no traj"

	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	with open(output_file, 'wb') as f:
		pickle.dump(paths_out, f)
	print("Saved to ", output_file, " ~!")
	return ""



if __name__ == '__main__':
	
	args = get_args()
	env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
	# override some parameters for testing
	env_cfg.env.num_envs = NUM_ENVS # min(env_cfg.env.num_envs, NUM_ENVS)
	env_cfg.terrain.num_rows = 5
	env_cfg.terrain.num_cols = 5
	env_cfg.terrain.curriculum = False
	env_cfg.noise.add_noise = NOISE # False
	env_cfg.domain_rand.randomize_friction = NOISE # False
	env_cfg.domain_rand.push_robots = NOISE # False

	env_cfg.commands.ranges.lin_vel_x = [0.3,0.7]
	env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
	env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

	if ZERO_VEL:
		env_cfg.commands.ranges.lin_vel_x = [0.2,0.5]
		env_cfg.commands.ranges.lin_vel_y = [0.0,0.15]
		env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.15]

	if VEL0_5:
		env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
		env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
		env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

	if VEL0_4:
		env_cfg.commands.ranges.lin_vel_x = [0.4,0.4]
		env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
		env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

	env_cfg.terrain.mesh_type = 'plane'

	# prepare environment
	env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
	failed_set = set()
	#????????????????????????????????????
	# for name in codename_list:
	# 	for rate in rate_list:
	# 		failed_set.add(play(args, env, train_cfg, name, rate))
	# for name in ["LBA", "RBA", "LFA"]:
	# 	failed_set.add(play(args, env, train_cfg, name, 0))
	# play(args, env, train_cfg)	#?????????????????????????????????????????????
	play(args, env, train_cfg, "LFA", 0)
	print(failed_set)
