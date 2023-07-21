import glob
import multiprocessing
import os
import pdb
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

# from d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from tqdm import tqdm

# from legged_gym.utils import Logger


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


# def get_d4rl_normalized_score(score, env_name):
#     env_key = env_name.split('-')[0].lower()
#     assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
#     return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


# def get_d4rl_dataset_stats(env_d4rl_name):
#     return D4RL_DATASET_STATS[env_d4rl_name]
def partial_traj(dataset_path_list, args, context_len=20, rtg_scale=1000, body_dim=12):
    """
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
    """
    big_list = []
    for pkl in tqdm(dataset_path_list):
        with open(pkl, "rb") as f:
            thelist = pickle.load(f)

        assert "body" in thelist[0]
        if args.cut == 0:
            big_list = big_list + thelist
        else:
            big_list = big_list + thelist[: args.cut]

    traj_dataset = D4RLTrajectoryDataset(
        big_list, context_len, rtg_scale, leg_trans_pro=True
    )
    assert body_dim == traj_dataset.body_dim

    state_mean, state_std = traj_dataset.get_state_stats(body=False)

    return traj_dataset, state_mean, state_std  # , body_mean, body_std

def evaluate_on_env_ppo_batch(
    model,
    device,
    context_len,
    env,
    rtg_target,
    rtg_scale,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
    render=False,
    prompt_policy=None,
):

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

        results["eval/avg_reward"] = np.sum(total_rewards) / num_eval_ep
        # results['eval/avg_reward'] = total_reward / num_eval_ep
        results["eval/avg_ep_len"] = total_timesteps / num_eval_ep

        return results


def flaw_generation(
    num_envs, bodydim=12, fixed_joint=[-1], flawed_rate=-1, device="cpu", upper_bound=1
):
    """
    num_envs: 环境数
    fixed_joint: 指定损坏的关节为fixed_joint(LIST) [0,11]，若不固定为-1
    flawed_rate: 损坏程度为flawed_rate, 若随机坏损为-1
    t(num_envs * len(fixed_joint)): 坏损的关节
    """
    if bodydim == 0:
        return None, None
    t = torch.randint(0, bodydim, (num_envs, 1))
    if -1 not in fixed_joint:
        t = torch.ones((num_envs, len(fixed_joint)), dtype=int) * torch.tensor(
            fixed_joint
        )
    bodys = torch.ones(num_envs, bodydim).to(device)
    for i in range(num_envs):
        for joint in [t[i]]:
            bodys[i, joint] = (
                random.random() * upper_bound if flawed_rate == -1 else flawed_rate
            )
    return bodys, t


def step_body(
    bodys, joint, rate=0.004, threshold=0, upper_bound = 1
):  # each joint has a flaw rate to be partial of itself.
    """
    joint: (num_envs, num) OR a single int, 每个环境对应的1个坏损关节
        #TODO: joint will become (num_envs, num), num means the number of flawed joints.
    rate: 每个step，有rate的概率使得关节扭矩往下掉，剩余扭矩比例随机
    threshold， 在剩余扭矩高于threshold时，重置到随机的一个扭矩。
    """
    num_envs = bodys.shape[0]
    t = torch.rand(num_envs)
    t = (t < rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodys.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodys.device)
        # print(bodys.shape, joint.shape, t.shape)
        p = torch.gather(bodys, 1, joint) * t
        bodys = torch.scatter(bodys, 1, joint, p)
        if threshold > 0:
            tmp = torch.gather(bodys, 1, joint)
            t = (tmp < threshold) * torch.rand(num_envs, device=bodys.device)
            t = t.to(bodys.device)
            t = 1 / (1 - t)
            bodys = torch.scatter(bodys, 1, joint, t * tmp)
            bodys = torch.min(bodys, torch.ones_like(bodys))
    else:
        bodys[:, joint] *= t
        if threshold > 0:  # Here we assume that joint must be a single int
            t = (bodys[:, joint] < threshold) * torch.rand(
                num_envs, device=bodys.device
            )
            t = t.to(bodys.device)
            t = 1 / (1 - t)
            bodys[:, joint] *= t
            bodys = torch.min(bodys, torch.ones_like(bodys))

    return bodys


def disable_leg(actions, target: str = "joint", index: int = 2):
    """用以让机器狗某条退或者某个关节失能，
    暂时只提供单关节失能用以初步测试

    Args:
        actions (_type_): 原动作
        target: 标识需要失能的目标类别
        index: 需要失能的具体目标

    Returns:
        _type_: 失能后动作
    """
    if target == "joint":
        actions[:, index] = 0  # 将指定索引的关节置0，暂定左前腿的2号关节失能
    elif target == "leg":
        actions[:, 3 * index : 3 * index + 3] = -1.0
    else:
        pass

    return actions

def evaluate_on_env_batch_distill(
    model,
    device,
    context_len,
    env,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
):
    """本函数用来测试强行蒸馏十二种步态的transformer表现如何"""
    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions

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
        actions = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        states = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )  # Here we assume that obs = state !!
        running_body, joints = flaw_generation(
            eval_batch_size, fixed_joint=[-1], device=device
        )
        running_body = running_body.to(device)
        running_state, _ = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        total_steps = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)

        for t in range(max_test_ep_len):
            states[:, t, :] = running_state
            states[:, t, :] = (states[:, t, :] - state_mean) / state_std

            if t < context_len:
                _, act_preds = model.forward(
                    timesteps[:, :context_len],
                    states[:, :context_len],
                    actions[:, :context_len],
                )

                act = act_preds[:, t].detach()
            else:
                _, act_preds = model.forward(
                    timesteps[:, t - context_len + 1 : t + 1],
                    states[:, t - context_len + 1 : t + 1],
                    actions[:, t - context_len + 1 : t + 1],
                )
                act = act_preds[:, -1].detach()
            running_state, _, running_reward, done, infos = env.step(
                act.cpu(), running_body
            )

            actions[:, t] = act
            total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            total_steps += dones == 0
            dones += done.detach().cpu().numpy()

            running_body = step_body(running_body, joints, rate=0.01)

    results["eval/avg_reward"] = np.mean(total_rewards)
    results["eval/avg_ep_len"] = np.mean(total_steps)
    return results


def evaluate_on_env_batch_body(
    model,
    device,
    context_len,
    env,
    body_target,
    rtg_scale = None,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
    body_mean=None,
    body_std=None,
    render=False,
    upper_bound = 0, 
    prompt_policy=None,
    body_pre:bool = False,
    body_gt:bool = True,
    body_pre_acu = False
):
    """
    用来测试EAT及其变种的模型表现
    ------------------
    model：待测试模型
    device：测试硬件号
    context_len：模型上下文长度
    env：测试环境
    body_target：目前除了用来提供bodydim外暂时无用
    rtg_scale：无用 待删除
    num_eval_ep=10：测试所用环境数量
    max_test_ep_len=1000：每个环境中测试长度
    state_mean=None：如其名
    state_std=None：如其名
    body_mean=None：如其名，无用
    body_std=None：如其名，无用
    render=False：是否渲染，暂时无用
    prompt_policy：暂时无用
    body_pre=False：是否将自身预测的body作为下一帧的body输入
    body_gt：是否将真实body作为下一帧预测的输入（body_pre为true时为body预测的输入，否则为action预测的输入）
    body_pre_acu: 是否返还bodypre准确度
    """

    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
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
        actions = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        states = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )  # Here we assume that obs = state !!
        running_body, joints = flaw_generation(
            eval_batch_size, bodydim=body_dim, fixed_joint=[-1], device=device, upper_bound=upper_bound
        )
        running_body = running_body.to(device)
        bodys = running_body.expand(max_test_ep_len, eval_batch_size, body_dim).type(
            torch.float32
        )
        bodys = torch.transpose(bodys, 0, 1).to(device)
        running_state, _ = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)
        total_pre = 0
        for t in range(max_test_ep_len):
            total_timesteps += 1
            states[:, t, :] = running_state
            states[:, t, :] = (states[:, t, :] - state_mean) / state_std
            if body_gt:#给予真实body的值
                bodys[:, t, :] = running_body
            if t < context_len:
                _, act_preds, _ = model.forward(
                    timesteps[:, :context_len],
                    states[:, :context_len],
                    actions[:, :context_len],
                    body=bodys[:, :context_len],
                )

                if prompt_policy is None:
                    act = act_preds[:, t].detach()
                else:
                    act = prompt_policy(running_state.to(device))
            else:
                if body_pre:
                    _, _, body_preds = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        body=bodys[:, t - context_len + 1 : t + 1],
                    )
                    bodys[:, t] = body_preds[:, -1].detach().clamp(0.0, 1.0)    #! 注意，这里为了增加actionpre的准确率所以进行了clamp，正常这一步也许再model内进行比较合理？
                    # body_preds = torch.nn.functional.softmax(body_preds,dim=-1)
                    # pre_target = torch.argmin(body_preds[:,-1], dim=-1).unsqueeze(1)    #! 注意，这里为了增加actionpre的准确率所以进行了clamp，正常这一步也许再model内进行比较合理？
                    # bodys[:, t] = torch.ones_like(body_preds[:,-1], dtype = torch.float32).scatter(
                    #     dim = -1,
                    #     index = pre_target,
                    #     src = torch.zeros_like(pre_target, dtype = torch.float32)
                    # )
                    # if((
                    #     torch.argmin(body_preds[:,10:], dim=-1).mode(dim = -1)[0] 
                    #      - torch.argmin(torch.nn.functional.softmax(running_body, dim=-1), dim=-1)
                    #      ).nonzero().shape[0] 
                    #     < running_body.shape[0]/10):    #预测与实际不符的情况小鱼十分之一时判据通过
                    #     total_pre += 1
                _, act_preds, body_preds = model.forward(
                    timesteps[:, t - context_len + 1 : t + 1],
                    states[:, t - context_len + 1 : t + 1],
                    actions[:, t - context_len + 1 : t + 1],
                    body=bodys[:, t - context_len + 1 : t + 1],
                )
                act = act_preds[:, -1].detach()
                if((    #! 这一部分body预测的eval问题很大，注意后续整改
                    torch.argmin(body_preds[:,10:], dim=-1).mode(dim = -1)[0]
                        - torch.argmin(torch.nn.functional.softmax(running_body, dim=-1), dim=-1)
                        ).nonzero().shape[0] 
                    < running_body.shape[0]/10):    #预测与实际不符的情况小鱼十分之一时判据通过
                        total_pre += 1
                
            running_state, _, running_reward, done, infos = env.step(
                act.cpu(), running_body
            )

            # if log_reward and infos["episode"]:
            #     num_episodes = torch.sum(env.reset_buf).item()
            #     if num_episodes > 0:
            #         logger.log_rewards(infos["episode"], num_episodes)

            actions[:, t] = act
            total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += (
                running_reward.detach().cpu().numpy()
                * (done == 0).detach().cpu().numpy()
            )
            dones += ~done.detach().cpu().numpy()

            running_body = step_body(running_body, joints, rate=0.004, upper_bound=upper_bound)

    results["eval/avg_reward"] = np.mean(total_rewards)
    results["eval/avg_ep_len"] = np.mean(dones)
    if body_pre_acu:
        results["eval/total_pre_rate"] = total_pre / (max_test_ep_len-20)
    return results

def evaluate_bodypre(
    model,
    run_model,
    device,
    context_len,
    env,
    eval_batch_size=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
):
    """
    这个函数写来纯纯粹粹为了测试torques预测准不准
    ------------------
    model：待测试模型
    test_model: 纯粹用来让机器狗走起来的
    device：测试硬件号
    context_len：模型上下文长度
    env：测试环境
    body_target：目前除了用来提供bodydim外暂时无用
    rtg_scale：无用 待删除
    num_eval_ep=10：测试所用环境数量
    max_test_ep_len=1000：每个环境中测试长度
    state_mean=None：如其名
    state_std=None：如其名
    """
    eval_batch_size = eval_batch_size  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    body_dim = 12

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
    run_model.eval()
    # logger = Logger(env.dt)

    with torch.no_grad():
        actions = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        states = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )  # Here we assume that obs = state !!

        bodys = torch.ones(
            (eval_batch_size, max_test_ep_len, body_dim),
            dtype=torch.float32,
            device=device,
        )
        state_mean, state_std =( run_model.state_mean).to(device), (run_model.state_std).to(device)
        
        running_state, _ = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)
        total_pre = 0
        # 创建形状为4096*12的全1张量，用于body状态
        running_bodies = torch.ones(eval_batch_size, body_dim, device=device)
        # 随机选择在第80~120步之间产生随机坏损的步骤
        faulty_step = torch.randint(80, 120, size=(eval_batch_size,), device=device)
        # 创建帕雷托分布对象，其中alpha是形状参数，scale是尺度参数           
        pareto_dist = torch.distributions.Pareto( scale= torch.tensor(0.05),  alpha= torch.tensor(2.0))#
        
        for t in range(max_test_ep_len):
            faulty_index = torch.randint(0,body_dim, (1,))
            mask = (t == faulty_step)
            if mask.any():
                # 生成从帕雷托分布中采样的随机数
                pareto_sample = pareto_dist.sample((sum(mask),)).clamp(0, 1.0).to(device)
                running_bodies[mask, faulty_index] = pareto_sample
            # 坏损生成完毕，进入主循环,更新obs与body值
            bodys[:,t,:] = running_bodies
            
            total_timesteps += 1
            states[:, t, :] = running_state
            states[:, t, :] = (states[:, t, :] - state_mean) / state_std

            start_t = 0 if t < context_len else t -context_len + 1 
            end_t = context_len if t < context_len else t + 1
            _, act_preds, _ = run_model.forward(
                                                    timesteps[:, start_t : end_t],
                                                    states[:, start_t : end_t],
                                                    actions[:, start_t : end_t],
                                                    body=bodys[:, start_t : end_t],
                                                )
            act = act_preds[:, t if t<context_len else -1].detach()
            
            states1 = states[:,start_t : end_t,:24]
            actions1 = (actions[:,start_t : end_t]  * env.action_scale + env.default_dof_pos.unsqueeze(dim=1).repeat(1, 20, 1))       # B x T x act_dim
            states_in = torch.cat((states1, actions1), dim=2)
            body_preds = model.forward(
                                            timesteps[:, start_t : end_t],
                                            states_in,
                                            actions[:, start_t : end_t],
                                            body=bodys[:, start_t : end_t]
                                        )
            score = 0
            if t>20:
                for i in range(body_dim):
                    score += ((body_preds[:,-1, i]<0.5) & (running_bodies[:,i] < 1)).sum()
                    score -= ((body_preds[:,-1, i]>0.5) & (running_bodies[:,i] < 1)).sum()
            
            actions[:, t] = act
            running_state, _, running_reward, done, infos = env.step(
                act.cpu(), bodys[:,-1]
            )           

            # total_reward += np.sum(score)
            total_rewards += (
                running_reward.detach().cpu().numpy()
                * (done == 0).detach().cpu().numpy()
            )
            dones += ~done.detach().cpu().numpy()
            
    results["eval/avg_reward"] = score
    results["eval/avg_ep_len"] = np.mean(dones)
    # if body_pre_acu:
    #     results["eval/total_pre_rate"] = total_pre / (max_test_ep_len-20)
    return results

def evaluate_torques(
    model,
    run_model,
    device,
    context_len,
    env,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
):
    """
    这个函数写来纯纯粹粹为了测试torques预测准不准
    ------------------
    model：待测试模型
    test_model: 纯粹用来让机器狗走起来的
    device：测试硬件号
    context_len：模型上下文长度
    env：测试环境
    body_target：目前除了用来提供bodydim外暂时无用
    rtg_scale：无用 待删除
    num_eval_ep=10：测试所用环境数量
    max_test_ep_len=1000：每个环境中测试长度
    state_mean=None：如其名
    state_std=None：如其名
    """
    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    body_dim = 12

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
    run_model.eval()
    # logger = Logger(env.dt)

    with torch.no_grad():
        actions = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        states = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )  # Here we assume that obs = state !!

        bodys = torch.ones(
            (eval_batch_size, context_len, body_dim),
            dtype=torch.float32,
            device=device,
        )
        
        running_state, _ = env.reset()

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)
        total_pre = 0
        
        for t in range(max_test_ep_len):
            total_timesteps += 1
            states[:, t, :] = running_state
            states[:, t, :] = (states[:, t, :] - state_mean) / state_std

            if t < context_len:
                _, act_preds, _ = run_model.forward(
                    timesteps[:, :context_len],
                    states[:, :context_len],
                    actions[:, :context_len],
                    body=bodys,
                )
                act = act_preds[:, t].detach()

            else:
                _, act_preds, _ = run_model.forward(
                    timesteps[:, t - context_len + 1 : t + 1],
                    states[:, t - context_len + 1 : t + 1],
                    actions[:, t - context_len + 1 : t + 1],
                    body=bodys,
                )
                act = act_preds[:, -1].detach()
                
            running_state, _, running_reward, done, infos = env.step(
                act.cpu(), bodys[:,-1]
            )

            actions[:, t] = act
            start_t = 0 if t < context_len else t -context_len + 1 
            end_t = context_len if t < context_len else t + 1
            torqs_preds = model.forward(
                                            timesteps[:, start_t : end_t],
                                            states[:, start_t : end_t],
                                            actions[:, start_t : end_t],
                                            # body=body
                                        )
            total_bias = torqs_preds[:,-1].cpu().detach().numpy() - env.torques.cpu().detach().numpy()
        
            total_reward += np.sum(total_bias)
            total_rewards += (
                running_reward.detach().cpu().numpy()
                * (done == 0).detach().cpu().numpy()
            )
            dones += ~done.detach().cpu().numpy()
            
    results["eval/avg_reward"] = 1/np.mean(total_rewards)
    results["eval/avg_ep_len"] = np.mean(dones)
    # if body_pre_acu:
    #     results["eval/total_pre_rate"] = total_pre / (max_test_ep_len-20)
    return results

def parallel_load(path):
    start = time.time()
    print("START LOADING ......")

    def process(file):
        with open(file, "rb") as f:
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
    p.join()  # Wait for all child processes to close.

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
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        data_list.append(dataset)

    # pdb.set_trace()
    return data_list


class D4RLTrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        context_len,
        leg_trans=False,
        leg_trans_pro=False,
        bc=False,
    ):

        self.context_len = context_len

        # load dataset
        if type(dataset_path) == str:
            if dataset_path.endswith(".pkl"):
                with open(dataset_path, "rb") as f:
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
        for traj in self.trajectories:  #每个traj都是单个环境的一条数据,self.trajectories是并列拼在一起的一条数据长度不一定相同
            traj_len = traj["observations"].shape[0]    #该环境收集数据的长度
            min_len = min(min_len, traj_len)    #一时之间没太明白为啥有个10**6的上限，怕轨迹太长吗
            states.append(traj["observations"]) #将每个环境的单条数据中的obs拼接在一起
            traj["returns_to_go"] = traj["body"]
            bodies.append(traj["body"])     #将每个环境的单条数据中的body拼接在一起

        # used for input normalization
        states = np.concatenate(states, axis=0)         #这一步不会将state拼接吗，不会，真就只用在归一化上
        # bodies = np.concatenate(bodies, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        # normalize states
        for traj in self.trajectories:#! 再load时已经做过了
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

        if leg_trans_pro:
            pass
            # self.body_mean, self.body_std = np.mean(bodies, axis=0), np.std(bodies, axis=0) + 1e-6
            # traj['returns_to_go'] = (traj['returns_to_go'] - self.body_mean) / self.body_std

    def get_state_stats(self, body=False):
        if body:
            pass
            # return self.state_mean, self.state_std, self.body_mean, self.body_std
        return self.state_mean, self.state_std

    @property
    def body_dim(self):
        return self.trajectories[0]["body"].shape[-1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx): #任选一条轨迹
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj["observations"][si : si + self.context_len])
            actions = torch.from_numpy(traj["actions"][si : si + self.context_len])
            returns_to_go = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])), dtype=states.dtype
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])), dtype=actions.dtype
                    ),
                ],
                dim=0,
            )

            returns_to_go = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )
            # pdb.set_trace()

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return timesteps, states, actions, returns_to_go, traj_mask


class D4RLTrajectoryDatasetForTert(Dataset):
    def __init__(
        self, dataset_path, context_len, leg_trans_pro=False
    ):

        self.context_len = context_len

        # load dataset
        if type(dataset_path) == str:
            if dataset_path.endswith(".pkl"):
                with open(dataset_path, "rb") as f:
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
            traj_len = traj["observations"].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj["observations"])
            traj["returns_to_go"] = traj["body"]
            bodies.append(traj["body"])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        bodies = np.concatenate(bodies, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        # normalize states
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

        if leg_trans_pro:
            self.body_mean, self.body_std = (
                np.mean(bodies, axis=0),
                np.std(bodies, axis=0) + 1e-6,
            )

    def get_state_stats(self, body=False):
        if body:
            return self.state_mean, self.state_std, self.body_mean, self.body_std
        return self.state_mean, self.state_std

    @property
    def body_dim(self):
        return self.trajectories[0]["body"].shape[-1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj["observations"][si : si + self.context_len])
            actions = torch.from_numpy(traj["actions"][si : si + self.context_len])
            teacher_actions = torch.from_numpy(
                traj["teacher_actions"][si : si + self.context_len]
            )
            returns_to_go = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])), dtype=states.dtype
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])), dtype=actions.dtype
                    ),
                ],
                dim=0,
            )

            teacher_actions = torch.from_numpy(traj["teacher_actions"])
            teacher_actions = torch.cat(
                [
                    teacher_actions,
                    torch.zeros(
                        ([padding_len] + list(teacher_actions.shape[1:])),
                        dtype=teacher_actions.dtype,
                    ),
                ],
                dim=0,
            )

            returns_to_go = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )
            # pdb.set_trace()

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return timesteps, states, actions, teacher_actions, returns_to_go, traj_mask
    
    
param_dict = {}
param_dict_fid7 = {}
i_magic = 104
for i in np.arange(0.1, 0.35, 0.05):
    for j in np.arange(0.1, 0.35, 0.05):
        for k in np.arange(0.1, 0.35, 0.05):
            param_dict[i_magic] = [round(i, 2), round(j, 2), round(k, 2)]
            i_magic += 1
i_magic = 229
for i in [0.35, 0.4]:
    for j in np.arange(0.1, 0.35, 0.05):
        for k in np.arange(0.1, 0.35, 0.05):
            param_dict[i_magic] = [round(i, 2), round(j, 2), round(k, 2)]
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

    if dataset == "none":  # 测试正常模型机器人能否在EAT上运作
        datafile = "P20F10000-vel0.5-v0"
        i_magic_list = ["none"]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"

    if dataset == "IPPOtest":
        datafile = "Trajectory_IPPO"
        i_magic_list = [f"PPO_I_{x}" for x in range(2)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO":
        datafile = "Trajectory_IPPO"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO2":
        datafile = "Trajectory_IPPO_2"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO3":
        datafile = "Trajectory_IPPO_3"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO6":
        datafile = "Trajectory_IPPO_6"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO8":
        datafile = "Trajectory_IPPO_8"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO9":
        datafile = "Trajectory_IPPO_9"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPOUB3":
        datafile = "Trajectory_IPPO_UB3"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPOUB3_2":
        datafile = "Trajectory_IPPO_UB3_2"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody":
        datafile = "Trajectory_Ebody"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody2":
        datafile = "Trajectory_Ebody2"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody3":
        datafile = "Trajectory_Ebody3"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody4":
        datafile = "Trajectory_Ebody4"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "AMP":
        datafile = "Trajectory_AMP"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Short":
        datafile = "Trajectory_Short_0"
        i_magic_list = [f"ShortTraj_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "faulty":
        datafile = "P20F10000-vel0.5-v0"
        i_magic_list = [
            "none",
            "LFH",
            "LFK",
            "RFH",
            "RFK",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"
    if dataset == "continue1000_faulty":  # 包含连续torques表现不好的情况但数据量更少（不需要切分可以直接学
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in [
            "LFH",
            "LFK",
            "LFA",
            "RFH",
            "RFK",
            "RFA",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "continue1000_faultypro":  # 包含连续torques表现不好的情况但数据量更少的情况并且掺入大量0力矩的情况
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in [
            "LFH",
            "LFK",
            "LFA",
            "RFH",
            "RFK",
            "RFA",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        for name in ["LFA+", "RFA+", "RBA+", "LBA+"]:
            i_magic_list.append(f"{name}-0")
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "continue1000_mix":  # 包含连续torques表现不好的情况但数据量更少的情况并且掺入大量0力矩的情况
        datafile = "F1000-v0"
        i_magic_list = ["none-1"]
        for name in [
            "LFH",
            "LFK",
            "LFA",
            "RFH",
            "RFK",
            "RFA",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]:
            # for rate in [0, 0.25]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        for name in ["LFAI", "RFAI", "RBAI", "LBAI"]:
            i_magic_list.append(f"{name}-0")
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "TertEAT1024":  # Tert的情况
        datafile = "F1024-v0"
        i_magic_list = ["none-1"]
        for name in [
            "LFH",
            "LFK",
            "LFA",
            "RFH",
            "RFK",
            "RFA",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]:
            for rate in [0, 0.25, 0.5, 0.75]:
                i_magic_list.append(f"{name}-{rate}")
        eval_body_vec = [1 for _ in range(12)]

    # ================================================================================
    if dataset == "top1000":
        datafile = "1000outof10000-vel0.5-v0"
        i_magic_list = [
            "none",
            "LFH",
            "LFK",
            "RFH",
            "RFK",
            "LBH",
            "LBK",
            "LBA",
            "RBH",
            "RBK",
            "RBA",
        ]
        eval_body_vec = [1 for _ in range(12)]
        eval_env = "none"

    return datafile, i_magic_list, eval_body_vec, eval_env
