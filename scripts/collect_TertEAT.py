import argparse
import os
import pickle
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import random
import csv
from datetime import datetime

import numpy as np

from legged_gym.envs import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import D4RLTrajectoryDataset, evaluate_on_env,  evaluate_on_env_batch_body, get_dataset_config #, get_d4rl_normalized_score,
from model import DecisionTransformer, LeggedTransformer, LeggedTransformerPro, MLPBCModel
from legged_gym.utils import  get_args
from legged_gym.utils.task_registry_embody import  task_registry
import wandb
from singlea1 import A1

from tqdm import trange, tqdm

SAVE_DIR = os.path.join(parentdir, "data/Tert_data")
NUM_ENVS = 7000 #10000 # 400 #4000 #1000 # 50# 20000 #
REP = 1 #10 #20

# param_dict = {}
## 断腿版本的i_magic命名规则：共12维，1代表正常 小数代表力矩折扣率
## 12维依次是左前右前左后右后腿 从上到下的关节顺序
## 注释中代号：L为左 R为右 F为前 B为后 H为臀 K为膝 A为踝 出现字母代表对应位置坏损
codename_list = []	#存储每条腿的字母代号
for i in ["F", "B"]:
	for j in ["L", "R"]:
		for k in ["H", "K", "A"]:
			codename_list.append(j+i+k)
rate_list = [0, 0.25, 0.5, 0.75]#储存现有的故障比率模型

def flaw_generation(num_envs, rate=1, bodydim = 12, fixed_joint = -1): # rate: the rate of env which have a flawed joint
	# import pdb
	# pdb.set_trace()
	import random
	t = torch.rand(num_envs)
	p = torch.randint(1, bodydim+1, (num_envs,))
	if not fixed_joint == -1:
		p = fixed_joint + 1
	t = (t<rate) * p
	bodys = torch.ones(num_envs, bodydim)
	for i in range(num_envs):
		if t[i] > 0:
			bodys[i][t[i]-1] = random.random()
	# print(bodys[:10,:])
	print(bodys.shape)
	return bodys

def flaw(bodys, joint, rate = 0.004, threshold = 0.4, descend = 0.005): #each joint has a flaw rate to be partial of itself.
	num_envs = bodys.shape[0]
	t = torch.rand(num_envs)
	import random
	t = (t<rate) * random.random()
	t = 1 - t
	t = t.to(bodys.device)
	# print(t.shape)
	bodys[:, joint] *= t
	for i in range(num_envs):
		if bodys[i,joint] > threshold:
			bodys[i,joint] -= descend
	# print(bodys[:10,:])
	return bodys



def play(args, env, train_cfg, fault_type = "none"):
	"""
	用以采集某一种情况下得Tert_EAT数据
	Args:
		args (Namespace): 一些参数
		env (envClass): 可复用环境
		train_cfg (_type_): 算法配置
		fault_type (str, optional): 坏损的腿的代号，none表示完好 . Defaults to "none".
		fault_rate (int, optional): 坏损程度，小于1 1代表完好. Defaults to 1.
	"""
	print(f"collecting {fault_type} ...")
	#preparing work
	env.reset()
	data_set = {'observations':[], 'bodys':[], 'actions':[], 'teacher_actions':[], 'terminals':[]}
	total_dones = np.zeros(NUM_ENVS)
	total_rewards = np.zeros(NUM_ENVS)
	#preparing output dir
	ver_index = 0
	output_file = f"{SAVE_DIR}/{fault_type}.pkl"
	while os.path.isfile(output_file):
		ver_index += 1
		output_file = f"{SAVE_DIR}/{fault_type}.pkl"
	print("Preparing file to ", output_file)
 
	#loading ppo model
	train_cfg.runner.resume = True
	train_cfg.runner.load_run = "PPO_Models"
	fault_id = codename_list.index(fault_type)

	train_cfg.runner.checkpoint = fault_id
	# model_root = os.path.join(parentdir, "models")
	# if fault_type != "none":	#存在故障的模型分在不同文件及中
	# 	train_cfg.runner.load_run = fault_type + "_" + str(fault_rate)
	# 	model_root = os.path.join(model_root, str(fault_rate) + "_torques")
	# 	#判断模型文件是否存在 若不存在则报错弹出
	# 	if not os.path.exists(os.path.join(model_root,train_cfg.runner.load_run)):
	# 		print(f"no model file{fault_type}_{fault_rate}")
	# 		return fault_type + "_" + str(fault_rate) + " file not exists"
	# ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=model_root)
	# policy = ppo_runner.get_inference_policy(device=env.device)
	ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
	policy = ppo_runner.get_inference_policy(device=env.device)

	#loading EAT model
	# loading pre_record stds,means...
	model_path = os.path.join(parentdir, "EAT_runs/EAT_IPPO2_01/")
	state_mean, state_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy")
	# loading EAT model
	eval_batch_size = NUM_ENVS  # envs
	max_test_ep_len = int(env.max_episode_length)    	#iters
	
	state_dim = 48
	act_dim = 12
	body_dim = 12	

	context_len = 20      # K in decision transformer
	n_blocks = 6            # num of transformer blocks
	embed_dim = 128          # embedding (hidden) dim of transformer 
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
			state_std=state_std
            ).to(device)
	EAT_model.load_state_dict(torch.load(
        os.path.join(model_path,"model_best.pt")
    ))
	EAT_model.eval()

	# collecting data
	with torch.no_grad():
		body_dim = 12
		state_mean = torch.from_numpy(state_mean).to(device)
		state_std = torch.from_numpy(state_std).to(device)

		running_body = flaw_generation(eval_batch_size, rate = 1, bodydim = body_dim, fixed_joint=fault_id)
		running_body = running_body.to(device)
		bodies = running_body.expand(max_test_ep_len, eval_batch_size, body_dim).type(torch.float32)
		bodies = torch.transpose(bodies, 0, 1).to(device)
		
		#recording data   
		# assert int(env.max_episode_length) == 1000
		timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
		timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
		actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
									dtype=torch.float32, device=device)
		states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
							dtype=torch.float32, device=device)

		obs = env.get_observations()
		for t in trange(REP*int(env.max_episode_length)):
			#以下循环在REP不为1的时候也许存在问题 懒得改了 以后有空再说吧  
			states[:,t,:] = obs
			states[:,t,:] = (states[:,t,:] - state_mean) / state_std

			bodies[:,t,:] = running_body
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
						
			teacher_actions = policy(obs.detach(), running_body)

			data_set['observations'].append(obs.cpu().detach().numpy()[:,:48])
			data_set['bodys'].append(running_body.cpu().detach().numpy())
			data_set['actions'].append(act.cpu().detach().numpy())
			data_set['teacher_actions'].append(teacher_actions.cpu().detach().numpy())

			obs, _, rews, dones, infos = env.step(actions[:,t,:].detach(), running_body)

			data_set['terminals'].append(dones.cpu().detach().numpy())		
			# data_set['next_observations'].append(obs.cpu().detach().numpy()[:,:48])
			running_body = flaw(running_body, joint = fault_id)

			total_rewards += rews.detach().cpu().numpy() * (total_dones == 0)
			total_dones += dones.detach().cpu().numpy()

	print("MEAN SCORE: ", np.mean(total_rewards))

	#recording data
	print("[REORGANISING DATA ......]")

	keys = ['observations', 'actions', 'teacher_actions', 'terminals']

	for k in keys:
		print("Preprocessing ", k)
		data_set[k] = np.array(data_set[k])


	obss = np.array(data_set['observations']).transpose((1,0,2))
	# nobss = np.array(data_set['next_observations']).transpose((1,0,2))
	bodys = np.array(data_set['bodys']).transpose((1,0,2))

	acts = np.array(data_set['actions']).transpose((1,0,2))
	teacher_acts = np.array(data_set['teacher_actions']).transpose((1,0,2))
	ds = np.array(data_set['terminals']).transpose()

	paths = []
	for obs_p, bodys_p, act_p, teacher_p, done_p in zip(obss, bodys, acts, teacher_acts, ds):
		obs_list = []
		bodys_list = []
		# nobs_list = []
		act_list = []
		teacher_act_list = []
		done_list = []
		path_dict = {}

		for obs_t, bodys_t, act_t, teacher_t, done_t in zip(obs_p, bodys_p, act_p, teacher_p, done_p):
			obs_list.append(obs_t)
			bodys_list.append(bodys_t)
			act_list.append(act_t)
			teacher_act_list.append(teacher_t)
			done_list.append(done_t)
			if done_t:
				break

		path_dict['observations'] = np.array(obs_list)
		# path_dict['next_observations'] = np.array(nobs_list)
		path_dict['body'] = np.array(bodys_list)
		path_dict['actions'] = np.array(act_list)
		path_dict['teacher_actions'] = np.array(teacher_act_list)
		path_dict['terminals'] = np.array(done_list)

		# embodiment = [1 for _ in range(12)]
		# if fault_type != "none":
		# 	embodiment[codename_list.index(fault_type)] = fault_rate #将坏损关节的embody置为坏损率
		# path_dict["body"] = np.tile(np.array(embodiment), (np.shape(path_dict["observations"])[0], 1))

		paths.append(path_dict)

	num_samples = np.sum([p['teacher_actions'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')

	# pdb.set_trace()
	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return fault_type + "no traj"

	print("-->")

	with open(output_file, 'wb') as f:
		pickle.dump(paths, f)
	print(f"joint{fault_type} collection over.")
	print("Saved to ", output_file, " ~!")

	return ""

if __name__ == "__main__":
	#=========================================================
	#preparing env and args
	args = get_args()
	env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
	# override some parameters for testing
	env_cfg.env.num_envs = NUM_ENVS
	env_cfg.terrain.num_rows = 5
	env_cfg.terrain.num_cols = 5
	env_cfg.terrain.curriculum = False
	env_cfg.noise.add_noise = False # False
	env_cfg.domain_rand.randomize_friction = False # False
	env_cfg.domain_rand.push_robots = False # False
	
	env_cfg.commands.ranges.lin_vel_x = [0.03,0.7]
	env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
	env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]
	
	env_cfg.terrain.mesh_type = 'plane'
	
	# prepare environment
	env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
	failed_set = set()
	for name in codename_list:
		play(args, env, train_cfg, name)
	# play(args, env, train_cfg)	#采集四条腿都能用的机器狗的数据
	# play(args, env, train_cfg, "RBK", 0)
	# print(failed_set)