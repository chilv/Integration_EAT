import glob
import multiprocessing
import os
import pdb
import random
import time
import pickle
from regex import F
import torch
import numpy as np
from torch.utils.data import Dataset
# from d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
import imageio
from tqdm import tqdm
# from legged_gym.utils import Logger

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


# def get_d4rl_normalized_score(score, env_name):
#     env_key = env_name.split('-')[0].lower()
#     assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
#     return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


# def get_d4rl_dataset_stats(env_d4rl_name):
#     return D4RL_DATASET_STATS[env_d4rl_name]
def partial_traj(dataset_path_list, args, context_len=20, rtg_scale=1000, body_dim=12):
    '''
    当轨迹过长的时候，为照顾cpu运算负荷需要将数据集分几段加载，此函数处理一段，可以通过简单的复用调用来完成
    -------------
    输入：
    dataset_path_list: list of trajs
        一组轨迹的存储路径
    batch_size: int
        training batch size
    context_len: int
        transformer需要读取的上下文长度 
    rtg_scale: int 
        normalize returns to go
    body_dim: int
        number of body parts
    
    
    输出：
    traj_data_loader：DataLoader objects
        用以学习的轨迹对象
    state_mean,state_std: float
        状态值的均值和方差
    body_mean,body_std: float
        body值的均值方差
    '''
    big_list = []
    for pkl in tqdm(dataset_path_list):  
        with open(pkl, 'rb') as f:
            thelist = pickle.load(f)

        assert "body" in thelist[0]
        if args.cut == 0:
            big_list = big_list + thelist
        else:
            big_list = big_list + thelist[:args.cut]

    traj_dataset = D4RLTrajectoryDataset(big_list, context_len, rtg_scale, leg_trans_pro=True)
    assert body_dim == traj_dataset.body_dim
    
    state_mean, state_std = traj_dataset.get_state_stats(body=False)
    
    return traj_dataset, state_mean, state_std#, body_mean, body_std

def evaluate_on_env_ppo(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, prompt_policy=None):

    model.eval()
    results = {}
    total_reward = 0
    total_timesteps = 0

    with torch.no_grad():

        for _ in range(num_eval_ep):

            running_state = env.reset()
            # TESTING
            # running_state[:3] = [0.1,0,0]
            running_reward = 0

            for t in range(max_test_ep_len):
    
                total_timesteps += 1
                act = prompt_policy(torch.tensor(running_state).unsqueeze(0)).squeeze()

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                total_reward += running_reward

                if done:
                    break

        results['eval/avg_reward'] = total_reward / num_eval_ep
        results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

        return results

def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, prompt_policy=None, leg_trans=False):

    eval_batch_size = 1  # required for forward pass

    if leg_trans:
        assert rtg_scale == 1

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)
    

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()
            # TESTING
            # running_state[:3] = [0.1,0,0]
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            gif_images = []

            for t in range(max_test_ep_len):

                total_timesteps += 1
                # pdb.set_trace()
                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std
                # pdb.set_trace()

                # calcualate running rtg and add it in placeholder
                if not leg_trans:
                    running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg
                # pdb.set_trace()

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    # act = act_preds[0, t].detach()
                    if prompt_policy is None:
                        act = torch.zeros(12) # DEBUG!!!
                    else:
                        act = prompt_policy(torch.tensor(running_state).unsqueeze(0)).squeeze()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                # TESTING
                # running_state[:3] = np.zeros(3)
                # running_state[:3] = [0.1,0,0]
                # print(f"STEP{t}!")

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward
                # print("total_reward:", total_reward)

                if render:
                    # env.render()
                    gif_images.append(env.render(mode='rgb_array'))
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    # pdb.set_trace()

    if render:
        imageio.mimsave("play.gif", gif_images)

    return results

def evaluate_on_env_ppo_batch(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, prompt_policy=None):

    # model.eval()
    eval_batch_size = num_eval_ep
    results = {}
    total_reward = 0
    total_timesteps = 0

    total_rewards = np.zeros(eval_batch_size)
    dones = np.zeros(eval_batch_size)

    with torch.no_grad():

        # for _ in range(num_eval_ep):

        running_state = env.reset()
        # TESTING
        # running_state[:3] = [0.1,0,0]
        running_reward = 0

        for t in range(max_test_ep_len):

            total_timesteps += 1
            act = prompt_policy(running_state.to(device))

            running_state, running_reward, done, _ = env.step(act)

            total_reward += running_reward
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            dones += done.detach().cpu().numpy()

            if torch.all(done):
                break

        results['eval/avg_reward'] = np.sum(total_rewards) / num_eval_ep
        # results['eval/avg_reward'] = total_reward / num_eval_ep
        results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

        return results

def evaluate_on_env_batch(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, prompt_policy=None, leg_trans=False):

    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        if True:

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()
            # running_reward = 0
            running_reward = torch.zeros((eval_batch_size, ),
                                dtype=torch.float32, device=device)
            # running_rtg = rtg_target*np.ones((eval_batch_size,), dtype=np.float16) / rtg_scale

            gif_images = []
            total_rewards = np.zeros(eval_batch_size)
            dones = np.zeros(eval_batch_size)

            for t in range(max_test_ep_len):

                total_timesteps += 1
                # pdb.set_trace()
                # add state in placeholder and normalize
                # states[0, t] = torch.from_numpy(running_state).to(device)
                # states[0, t] = (states[0, t] - state_mean) / state_std
                states[:,t,:] = running_state
                states[:,t,:] = (states[:,t,:] - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                # running_rtg = running_rtg - (running_reward / rtg_scale)
                # rewards_to_go[0, t] = running_rtg
                try:
                    rewards_to_go[:, t] = rewards_to_go[:, t] - (torch.unsqueeze(running_reward.to(device), -1) / rtg_scale)
                except RuntimeError:
                    pdb.set_trace()

                # pdb.set_trace()
                if leg_trans:
                    rewards_to_go[:, t] = torch.ones((eval_batch_size, 1),
                                                      dtype=torch.float32, device=device)*rtg_target

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    # act = act_preds[0, t].detach()
                    if prompt_policy is None:
                        act = act_preds[:, t].detach()
                    else:
                        # act = prompt_policy(torch.tensor(running_state).unsqueeze(0)).squeeze()
                        act = prompt_policy(running_state.to(device))
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    # act = act_preds[0, -1].detach()
                    act = act_preds[:, -1].detach()


                # running_state, running_reward, done, _ = env.step(act.cpu().numpy())
                running_state, running_reward, done, _ = env.step(act.cpu())
                # pdb.set_trace()
                # print(f"STEP{t}!")

                # add action in placeholder
                # actions[0, t] = act
                actions[:, t] = act

                # total_reward += running_reward
                total_reward += np.sum(running_reward.detach().cpu().numpy())
                total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
                dones += done.detach().cpu().numpy()
                # print("SCORE: ", total_reward)
                # print("="*30)

                # if render:
                #     # env.render()
                #     gif_images.append(env.render(mode='rgb_array'))
                if torch.all(done):
                    break

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.sum(total_rewards) / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps
    
    # pdb.set_trace()

    # if render:
    #     imageio.mimsave("play.gif", gif_images)

    return results

def flaw_generation(num_envs, bodydim = 12, fixed_joint = [-1], flawed_rate=1, device = "cpu"):
    '''
        num_envs: 环境数
        fixed_joint: 指定损坏的关节为fixed_joint(LIST) [0,11]，若不固定为-1
        flawed_rate: 损坏程度为flawed_rate, 若随机坏损为-1
        t(num_envs * len(fixed_joint)): 坏损的关节
    '''
    if bodydim == 0:
        return None, None
    t = torch.randint(0, bodydim, (num_envs,1))
    if -1 not in fixed_joint:
        t = torch.ones((num_envs, len(fixed_joint)), dtype=int) * torch.tensor(fixed_joint)
    bodys = torch.ones(num_envs, bodydim).to(device)
    for i in range(num_envs):
        for joint in [t[i]]:
            bodys[i, joint] = random.random() if flawed_rate == -1 else flawed_rate
    return bodys, t
       
def step_body(bodys, joint, rate = 0.004, threshold = 0): #each joint has a flaw rate to be partial of itself.
    '''
    joint: (num_envs, num) OR a single int, 每个环境对应的1个坏损关节 
        #TODO: joint will become (num_envs, num), num means the number of flawed joints.
    rate: 每个step，有rate的概率使得关节扭矩往下掉，剩余扭矩比例随机
    threshold， 在剩余扭矩高于threshold时，重置到随机的一个扭矩。
    '''        
    num_envs = bodys.shape[0]
    t = torch.rand(num_envs)
    t = (t<rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodys.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodys.device)
        # print(bodys.shape, joint.shape, t.shape)
        p = torch.gather(bodys, 1, joint) * t
        bodys = torch.scatter(bodys, 1, joint, p)
        if threshold >0: 
            tmp = torch.gather(bodys, 1, joint)
            t = (tmp < threshold) * torch.rand(num_envs, device = bodys.device)
            t = t.to(bodys.device)
            t = 1/(1 - t)
            bodys = torch.scatter(bodys, 1, joint, t * tmp)
            bodys = torch.min(bodys, torch.ones_like(bodys))
    else:
        bodys[:, joint] *= t
        if threshold >0: # Here we assume that joint must be a single int
            t = (bodys[:,joint] < threshold) * torch.rand(num_envs, device = bodys.device)
            t = t.to(bodys.device)
            t = 1/(1 - t)
            bodys[:,joint] *= t
            bodys = torch.min(bodys, torch.ones_like(bodys))

    return bodys

def evaluate_on_env_batch_body(model, device, context_len, env, body_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None,
                    body_mean=None, body_std=None, render=False, prompt_policy=None, nobody=False):

    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    # state_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]
    body_dim = len(body_target)

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    # logger = Logger(env.dt)
    
    with torch.no_grad():
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device) # Here we assume that obs = state !!
        # bodys = torch.ones((eval_batch_size, max_test_ep_len, body_dim),
        #                         dtype=torch.float32, device=device)
        running_body , joints = flaw_generation(eval_batch_size, bodydim = body_dim, fixed_joint=[-1], device=device)
        running_body = running_body.to(device)
        bodys = running_body.expand(max_test_ep_len, eval_batch_size, body_dim).type(torch.float32)
        bodys = torch.transpose(bodys, 0, 1).to(device)
        running_state, _  = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)

        for t in range(max_test_ep_len):
            total_timesteps += 1
            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            bodys[:,t,:] = running_body

            if t < context_len:
                _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            body= bodys[:,:context_len])
                
                if prompt_policy is None:
                    act = act_preds[:, t].detach()
                else:
                    # act = prompt_policy(torch.tensor(running_state).unsqueeze(0)).squeeze()
                    act = prompt_policy(running_state.to(device))
            else:
                _, act_preds, body_preds = model.forward(timesteps[:, t-context_len+1:t+1],
                                        states[:, t-context_len+1:t+1],
                                        actions[:, t-context_len+1:t+1],
                                        body=bodys[:, t-context_len+1:t+1])
                act = act_preds[:, -1].detach()
            running_state, _, running_reward, done, infos = env.step(act.cpu(), running_body)
            
            # if log_reward and infos["episode"]:
            #     num_episodes = torch.sum(env.reset_buf).item()
            #     if num_episodes > 0:
            #         logger.log_rewards(infos["episode"], num_episodes)

            actions[:,t] = act
            total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            dones += done.detach().cpu().numpy()

            running_body = step_body(running_body, joints, rate = 0.04, threshold=0.4)   

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.sum(total_rewards) / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps    #! 这里timestep的记录方式有点问题，无法记录中途坠毁的情况，后续需要关注一下

    return results

def evaluate_with_env_batch_body(model, device, context_len, env, body_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None,
                    body_mean=None, body_std=None, render=False, prompt_policy=None, nobody=False):

    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    # state_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]
    body_dim = len(body_target)

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    # logger = Logger(env.dt)
    
    with torch.no_grad():
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device) # Here we assume that obs = state !!
        running_body , joints = flaw_generation(eval_batch_size, bodydim = body_dim, fixed_joint=[-1], device=device)
        running_body = running_body.to(device)
        bodys = running_body.expand(max_test_ep_len, eval_batch_size, body_dim).type(torch.float32)
        bodys = torch.transpose(bodys, 0, 1).to(device)
        running_state, _  = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)

        for t in range(max_test_ep_len):
            total_timesteps += 1
            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            # bodys[:,t,:] = running_body

            if t < context_len:
                _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            body= bodys[:,:context_len])
                
                if prompt_policy is None:
                    act = act_preds[:, t].detach()
                else:
                    # act = prompt_policy(torch.tensor(running_state).unsqueeze(0)).squeeze()
                    act = prompt_policy(running_state.to(device))
            else:
                _, act_preds, body_preds = model.forward(timesteps[:, t-context_len+1:t+1],
                                        states[:, t-context_len+1:t+1],
                                        actions[:, t-context_len+1:t+1],
                                        body=bodys[:, t-context_len+1:t+1])
                act = act_preds[:, -1].detach()
                bodys[:, t] = body_preds[:, -1].detach()
            running_state, _, running_reward, done, infos = env.step(act.cpu(), running_body)
            
            # if log_reward and infos["episode"]:
            #     num_episodes = torch.sum(env.reset_buf).item()
            #     if num_episodes > 0:
            #         logger.log_rewards(infos["episode"], num_episodes)

            actions[:,t] = act
            total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            dones += done.detach().cpu().numpy()

            running_body = step_body(running_body, joints, rate = 0.04, threshold=0.4)   

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.sum(total_rewards) / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps    #! 这里timestep的记录方式有点问题，无法记录中途坠毁的情况，后续需要关注一下

    return results

def evaluate_on_env_bc_batch(model, device, context_len, env, body_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None,
                    body_mean=None, body_std=None, render=False, prompt_policy=None, nobody=False):

    results = {}
    state_dim = env.observation_space.shape[0] + len(body_target)
    act_dim = env.action_space.shape[0]

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)

    model.eval()
    model.to(device=device)

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    assert body_target is not None
    body_dim = len(body_target)
    body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
    state_body = body_target.expand(num_eval_ep, body_dim).type(torch.float32)
    state = torch.cat([state_body, state], dim=1)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    # actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    # rewards = torch.zeros(0, device=device, dtype=torch.float32)
    # target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    states = torch.unsqueeze(state, 1)
    actions = torch.zeros((num_eval_ep, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((num_eval_ep, 0), device=device, dtype=torch.float32)
    target_return = torch.zeros((num_eval_ep, act_dim), device=device, dtype=torch.float32)

    episode_return, episode_length = 0, 0
    total_rewards = np.zeros(num_eval_ep)
    dones = np.zeros(num_eval_ep)

    for t in range(max_test_ep_len):

        # add padding
        # actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        actions = torch.cat([actions, torch.zeros((num_eval_ep, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((num_eval_ep, 1), device=device)], dim=1)

        # action = model.get_action_batch(
        #     (states.to(dtype=torch.float32) - state_mean) / state_std,
        #     # actions.to(dtype=torch.float32),
        #     # rewards.to(dtype=torch.float32),
        #     # target_return=target_return,
        # )
        states_ = (states.to(dtype=torch.float32) - state_mean) / state_std
        # pdb.set_trace()
        if states_.shape[1] < context_len:
            states_ = torch.cat(
                [torch.zeros((states_.shape[0], context_len-states_.shape[1], state_dim), 
                             dtype=torch.float32, device=states_.device), states_], dim=1)
        states_ = states_.to(dtype=torch.float32)
        _, actions_, _ = model.forward(states_)
        # print(actions_)

        actions[:,-1] = actions_[:,-1]
        # action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(actions_[:,-1])

        state = torch.cat([state_body, state], dim=1)
        # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        # states = torch.cat([states, cur_state], dim=0)
        cur_state = torch.unsqueeze(state, 1)
        states = torch.cat([states, cur_state], dim=1)
        if states.shape[1] > context_len:
            states = states[:,-context_len:]
        rewards[:,-1] = reward

        episode_return += reward
        episode_length += 1

        total_rewards += reward.detach().cpu().numpy() * (dones == 0)
        dones += done.detach().cpu().numpy()

        if torch.all(done):
            break

    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = episode_length

    return results


def parallel_load(path):
    start = time.time()
    print("START LOADING ......")

    def process(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    p = multiprocessing.Pool()
    res_list = []
    for f in glob.glob(os.path.join(path, "*.pkl")):
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        res = p.apply_async(process, [f]) 
        res_list.append(res)

    p.close()
    p.join() # Wait for all child processes to close.

    # print("======================")
    data_list = []
    for res in res_list:
        print("PROCESSING  ", res)
        data_list.append(res.get())
    end = time.time()
    print(f"FINISHED LOADING!! ({end - start}) SEC")
    # pdb.set_trace()
    return data_list

def load_path(path):
    print("START LOADING ......")

    data_list = []
    
    for file in tqdm(glob.glob(os.path.join(path, "*.pkl"))):
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        data_list.append(dataset)

    # pdb.set_trace()
    return data_list

class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale, leg_trans=False, leg_trans_pro=False, bc=False):

        self.context_len = context_len

        # load dataset
        if type(dataset_path) == str:
            if dataset_path.endswith(".pkl"):
                with open(dataset_path, 'rb') as f:
                    self.trajectories = pickle.load(f)
            else:
                self.trajectories = load_path(dataset_path)
        
        elif type(dataset_path) == list:
            self.trajectories = dataset_path
            
        if bc:
            print("CONCATENATE BODY INTO STATE ..........")
            for traj in tqdm(self.trajectories):
                traj['observations'] = np.concatenate([traj['body'], traj['observations']], axis=1)
                traj['next_observations'] = np.concatenate([traj['body'], traj['next_observations']], axis=1)
    
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        bodies = []
        # pdb.set_trace()
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            if not (leg_trans or leg_trans_pro):
                # calculate returns to go and rescale them
                traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
            elif leg_trans:
                traj['returns_to_go'] = traj['leg_length']
            elif leg_trans_pro:
                traj['returns_to_go'] = traj['body']
                bodies.append(traj['body'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        bodies = np.concatenate(bodies, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

        if leg_trans_pro:
            self.body_mean, self.body_std = np.mean(bodies, axis=0), np.std(bodies, axis=0) + 1e-6
            # traj['returns_to_go'] = (traj['returns_to_go'] - self.body_mean) / self.body_std


    def get_state_stats(self, body=False):
        if body:
            return self.state_mean, self.state_std, self.body_mean, self.body_std
        return self.state_mean, self.state_std

    @property
    def body_dim(self):
        return self.trajectories[0]['body'].shape[-1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)
            # pdb.set_trace()

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask


class D4RLTrajectoryDatasetForTert(Dataset):
    def __init__(self, dataset_path, context_len, leg_trans=False, leg_trans_pro=False, bc=False):

        self.context_len = context_len

        # load dataset
        if type(dataset_path) == str:
            if dataset_path.endswith(".pkl"):
                with open(dataset_path, 'rb') as f:
                    self.trajectories = pickle.load(f)
            else:
                self.trajectories = load_path(dataset_path)
        
        elif type(dataset_path) == list:
            self.trajectories = dataset_path
    
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        bodies = []
        # pdb.set_trace()
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            if not (leg_trans or leg_trans_pro):
                # calculate returns to go and rescale them
                # traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
                pass
            elif leg_trans:
                traj['returns_to_go'] = traj['leg_length']
            elif leg_trans_pro:
                traj['returns_to_go'] = traj['body']
                bodies.append(traj['body'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        bodies = np.concatenate(bodies, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

        if leg_trans_pro:
            self.body_mean, self.body_std = np.mean(bodies, axis=0), np.std(bodies, axis=0) + 1e-6
            # traj['returns_to_go'] = (traj['returns_to_go'] - self.body_mean) / self.body_std


    def get_state_stats(self, body=False):
        if body:
            return self.state_mean, self.state_std, self.body_mean, self.body_std
        return self.state_mean, self.state_std

    @property
    def body_dim(self):
        return self.trajectories[0]['body'].shape[-1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            teacher_actions = torch.from_numpy(traj['teacher_actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)
            
            teacher_actions = torch.from_numpy(traj['teacher_actions'])
            teacher_actions = torch.cat([teacher_actions,
                                torch.zeros(([padding_len] + list(teacher_actions.shape[1:])),
                                dtype=teacher_actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)
            # pdb.set_trace()

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, teacher_actions, returns_to_go, traj_mask



param_dict = {}
param_dict_fid7 = {}
i_magic = 104
for i in np.arange(0.1, 0.35, 0.05):
    for j in np.arange(0.1, 0.35, 0.05):
        for k in np.arange(0.1, 0.35, 0.05):
            param_dict[i_magic] = [round(i,2), round(j,2), round(k,2)]
            i_magic += 1
i_magic = 229
for i in [0.35, 0.4]:
    for j in np.arange(0.1, 0.35, 0.05):
        for k in np.arange(0.1, 0.35, 0.05):
            param_dict[i_magic] = [round(i,2), round(j,2), round(k,2)]
            i_magic += 1
i_magic = 279
for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    for j in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for k in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            param_dict_fid7[i_magic] = [i, j, k]
            i_magic += 1

inv_magic_dict = {repr(v): k for k, v in param_dict.items()}
inv_param_dict_fid7 = {repr(v): k for k, v in param_dict_fid7.items()}

def get_dataset_config(dataset):

    cut = 0
    eval_env = "none"
    datafile = ""
    i_magic_list = []
    eval_body_vec = []
    
    if dataset == "none":   #测试正常模型机器人能否在EAT上运作
        datafile = "P20F10000-vel0.5-v0"
        i_magic_list = ["none"]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"
    
    if dataset == "IPPOtest":
        datafile = "Trajectory_IPPO"
        i_magic_list = [f"PPO_I_{0}"]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO":
        datafile = "Trajectory_IPPO"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO2":
        datafile = "Trajectory_IPPO_2"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "faulty":
        datafile = "P20F10000-vel0.5-v0"
        i_magic_list = ["none", "LFH", "LFK", "RFH", "RFK", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"
    
    if dataset == "continue_faulty":    #包含连续torques表现不好的情况
        datafile = "F10000-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "continue4000_faulty":    #包含连续torques表现不好的情况但数据量更少（内存放不下
        datafile = "F4000-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-00")
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "continue1000_faulty":    #包含连续torques表现不好的情况但数据量更少（不需要切分可以直接学
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "continue1000_faultypro":    #包含连续torques表现不好的情况但数据量更少的情况并且掺入大量0力矩的情况
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        for name in ["LFA+", "RFA+", "RBA+", "LBA+"]:
            i_magic_list.append(f"{name}-0")
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "continue1000_mix":    #包含连续torques表现不好的情况但数据量更少的情况并且掺入大量0力矩的情况
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        for name in ["LFAI", "RFAI", "RBAI", "LBAI"]:
            i_magic_list.append(f"{name}-0")
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "TertEAT1024":    #Tert的情况
        datafile = "F1024-v0"
        i_magic_list = ["none-1"]
        for name in ["LFH", "LFK", "LFA", "RFH", "RFK", "RFA", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        eval_body_vec = [1 for _ in range(12)]
        
    #================================================================================
    if dataset == "top1000":
        datafile = "1000outof10000-vel0.5-v0"
        i_magic_list = ["none", "LFH", "LFK", "RFH", "RFK", "LBH", "LBK", "LBA", "RBH", "RBK", "RBA"]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"
    
    if dataset == "fid11223":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = [i for i in range(104, 229)]
        # eval_body_vec = [0.345, 0.2, 0.2]
        # eval_env = "a1magic101"
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid123":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.1, 0.2, 0.3]:
            for j in [0.1, 0.2, 0.3]:
                for k in [0.1, 0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid223":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.1, 0.2, 0.3]:
                for k in [0.1, 0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid22334":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid223-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.1, 0.2, 0.3]:
                for k in [0.1, 0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-1-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid22334-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid2334-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fidhyb0-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fidhyb1-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid223-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.1, 0.2, 0.3]:
                for k in [0.1, 0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-1-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid22334-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid2334-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fidhyb0-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fidhyb1-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid1122334-br":
        datafile = "P20F10000-vel0.4-rand-v0"
        i_magic_list = []
        for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid1122334-b":
        datafile = "P23F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid1122334":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-1":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])# 利用ijk三个参数来选去轨迹，但他们的意义分别是机器人可变部位的形状
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-1-5k":
        datafile = "P20F5000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid234-1-10k":
        datafile = "P20F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid24-1":
        datafile = "P20F1000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.4]:
            for j in [0.2, 0.3]:
                for k in [0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid24-1-5k":
        datafile = "P20F5000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.4]:
            for j in [0.2, 0.3]:
                for k in [0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid24-1-10k":
        datafile = "P20F10000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.4]:
            for j in [0.2, 0.3]:
                for k in [0.2, 0.3]:
                    i_magic_list.append(inv_magic_dict[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-223":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-234":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.1, 0.2, 0.3]:
                for k in [0.1, 0.2, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-234-1":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.3, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-22334":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-2334":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-hyb0":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.1, 0.15, 0.2]:
                for k in [0.2, 0.25, 0.3]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"

    if dataset == "fid7-hyb1":
        datafile = "T1000F4000-vel0.4-v0"
        i_magic_list = []
        for i in [0.25, 0.3, 0.35, 0.4]:
            for j in [0.2, 0.25, 0.3]:
                for k in [0.1, 0.15, 0.2]:
                    i_magic_list.append(inv_param_dict_fid7[repr([i, j, k])])
        eval_body_vec = [0.3, 0.2, 0.2]
        eval_env = "a1magic216"


    return datafile, i_magic_list, eval_body_vec, eval_env