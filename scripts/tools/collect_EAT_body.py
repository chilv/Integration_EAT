"""
本脚本用于收集已经可以运行的EAT，在不知body坏损的情况，生成的轨迹数据
主要意在：
	一方面收集更多完好情况下body突然坏损的迁移数据
 	另一方面利用ippo提升以外情况下agent的表现
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

from legged_gym.envs import *
from legged_gym.utils import task_registry, Logger, get_args
from model import LeggedTransformerBody, LeggedTransformerPro
import pdb
from tqdm import trange, tqdm
import pickle
from scripts.utils import flaw_generation, step_body


SAVE_DIR = os.path.join(os.path.dirname(parentdir), "data/Ebody3")
NUM_ENVS = 1024  # 10000 # 400 #4000 #1000 # 50# 20000 #EAT环境数不能开太高 显存受不了
REP = 1  # 10 #20


def pareto_body(bodies, joint, done, rate=0.05):
    # 模仿step_body，进行两点改动：
    # 1. 为了增加从1向任意值掉落的现象，每context_length步恢复置1，且期间只掉落一次
    # 2. 掉落概率符合帕雷特分布，为了将x隐射到0~1我采用tan函数，函数为：am^a/(tan(pi*x/2))^(a+1),取a=0.01,m=1
    """
    body： 当前时刻的body值
    joint: a single int, 对应的1个坏损关节
    rate: 每个step，有rate的概率使得关节扭矩往下掉，剩余扭矩比例随机
    """

    num_envs = bodies.shape[0]
    t = torch.rand((num_envs,1))
    t = ((t < rate) & (bodies[:, joint].detach().cpu() == 1.0)) * (
        1 - 0.01 / torch.tan(torch.pi * torch.rand(num_envs,1) / 2) ** 1.01
    )
    t = 1 - t
    t = t.to(bodies.device)
    bodies[:, joint] = t * bodies[:, joint] + done.unsqueeze(-1)
    # reset时bodies对应位置置一

    return bodies.clamp(0.0, 1.0)


def do_collect(args, env, ippo, eat_model, fault_tag=[-1]):
    """
    用以采集某一种情况下得Tert_EAT数据
    Args:
            args (Namespace): 一些参数
            env (envClass): 可复用环境
            ippo: ippo模型->貌似用不到？
            eat_model: eat模型
            fault_type (str, optional): 坏损的腿的代号, -1表示完好（默认）.
    Output:
            以文件的形式输出state, body, action
    """
    print(f"collecting {fault_tag}  ...")
    # preparing work
    env.reset()
    data_set = {
        "observations": [],
        "bodies_gt": [],
        "actions": [],
        "teacher_actions": [],
        "terminals": [],
    }

    # preparing output dir
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    ver_index = 1
    output_file = os.path.join(SAVE_DIR, f"EBody_v{ver_index}_{fault_tag[0]}.pkl")
    while os.path.isfile(output_file):
        ver_index += 1
        output_file = os.path.join(SAVE_DIR, f"EBody_v{ver_index}_{fault_tag[0]}.pkl")
    print("Preparing file to ", output_file)

    state_dim = args["state_dim"]
    body_dim = args["body_dim"]
    act_dim = args["act_dim"]
    state_mean = args["state_mean"]
    state_std = args["state_std"]

    with torch.no_grad():
        print("collecting...")
        timesteps = torch.arange(start=0, end=env.max_episode_length, step=1)
        timesteps = timesteps.repeat(NUM_ENVS, 1).to(device)
        actions = torch.zeros(
            (NUM_ENVS, env.max_episode_length, act_dim), dtype=torch.float32, device=device
        )
        states = torch.zeros(
            (NUM_ENVS, env.max_episode_length, state_dim),
            dtype=torch.float32,
            device=device,
        )
        body_target = torch.tensor(
            [1 for _ in range(body_dim)], dtype=torch.float32, device=device
        )
        bodies = body_target.expand(NUM_ENVS, env.max_episode_length, body_dim).type(
            torch.float32
        )

        running_state = env.reset()[0]
        running_body = bodies[:, 0, :].clone()
        done = torch.zeros(
            (NUM_ENVS, env.max_episode_length, 1), dtype=torch.float32, device=device
        )
        total_rew = 0

        # collecting data
        for t in trange(REP * int(env.max_episode_length)):
            # 循环在REP不为1的时候也许存在问题 懒得改了 以后有空再说吧
            if t < context_len:
                _, act_preds, _ = eat_model.forward(
                    timesteps[:, :context_len],
                    states[:, :context_len],
                    actions[:, :context_len],
                    body=bodies[:, :context_len],
                )
                # body = body_preds[:, t].detach()
                act = act_preds[:, t].detach()
            else:   # t >= context_len
                #generate body of this round
                running_body = pareto_body(
                    bodies=running_body, joint=fault_tag, done=done, rate=0.05
                )
                _, act_preds, _ = eat_model.forward(
                    timesteps[:, t - context_len + 1 : t + 1],
                    states[:, t - context_len + 1 : t + 1],
                    actions[:, t - context_len + 1 : t + 1],
                    body=bodies[:, t - context_len + 1 : t + 1],
                )
                act = act_preds[:, -1].detach()
                
            teacher_act = policy(states[:,t].detach(), running_body)        

            data_set["observations"].append(running_state.cpu().detach().numpy()[:, :48])
            data_set["bodies_gt"].append(running_body.cpu().detach().numpy())
            data_set["actions"].append(act.cpu().detach().numpy())
            data_set["teacher_actions"].append(teacher_act.cpu().detach().numpy())
            
            # prepare data of next round
            running_state, _, running_reward, done, infos = env.step(
                act.cpu(), running_body
            )
            data_set["terminals"].append(done.cpu().detach().numpy())
            actions[:, t] = act
            states[:, t] = (running_state - state_mean) / state_std            
            
            total_rew += running_reward #这里相当于每个agent都有无限次机会，不合理，但是因为此变量暂时无用，所以没改

    # recording data
    print("[REORGANISING DATA ......]")
    keys = ["observations", "bodies_gt", "actions", "terminals"]

    for k in keys:
        print("Preprocessing ", k)
        data_set[k] = np.array(data_set[k])

    obss = np.array(data_set["observations"]).transpose((1, 0, 2))
    acts = np.array(data_set["actions"]).transpose((1, 0, 2))
    teacher_acts = np.array(data_set["teacher_actions"]).transpose((1, 0, 2))
    bodies_gts = np.array(data_set["bodies_gt"]).transpose((1, 0, 2))
    ds = np.array(data_set["terminals"]).transpose()

    paths = []
    # for obs_p, act_p, body_p, done_p in zip(obss, acts, bodies_gts, ds):
    for obs_p, act_p, tact_p, body_p, done_p in zip(obss, acts, teacher_acts, bodies_gts, ds):
        obs_list = []
        act_list = []
        tact_list = []
        body_list = []
        done_list = []
        path_dict = {}

        # for obs_t, act_t, body_t, done_t in zip(obs_p, act_p, body_p, done_p):
        for obs_t, act_t, tact_t, body_t, done_t in zip(obs_p, act_p, tact_p, body_p, done_p):
            obs_list.append(obs_t)
            act_list.append(act_t)
            tact_list.append(tact_t)
            body_list.append(body_t)
            done_list.append(done_t)
            # 在当前的设定下由于会频繁出现done的情况，所以注释以下语句以防采集数据过短且分布不统一
            # if done_t:
            #     break

        path_dict["observations"] = np.array(obs_list)
        path_dict["actions"] = np.array(act_list)
        path_dict["teacher_actions"] = np.array(tact_list)
        path_dict["body"] = np.array(body_list)
        path_dict["terminals"] = np.array(done_list)

        paths.append(path_dict)

    num_samples = np.sum([p["actions"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")

    # pdb.set_trace()
    if num_samples == 0:
        print("NO USEFUL TRAJECTORIES !")
        return "no traj of joint {fault_tag}"

    print("-->")

    with open(output_file, "wb") as f:
        pickle.dump(paths, f)
    print(f"joint {fault_tag}'s collection over.")
    print("Saved to ", output_file, " ~!")

    return ""


if __name__ == "__main__":
    # =========================================================
    # preparing args
    with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)

    device = torch.device(args["device"])  # setting flexible

    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args["body_dim"]

    context_len = args["context_len"]  # K in decision transformer
    n_blocks = args["n_blocks"]  # num of transformer blocks
    embed_dim = args["embed_dim"]  # embedding (hidden) dim of transformer
    n_heads = args["n_heads"]  # num of transformer heads
    dropout_p = args["dropout_p"]  # dropout probability
    model_name = "EAT_given_body_IPPO_02"
    # =========================================================

    # =========================================================
    # prepare enviornment
    print("Init envs...")
    env_args = get_args()
    env_args.sim_device = args["device"]

    env_cfg, train_cfg = task_registry.get_cfgs(name=args["task"])
    # override some parameters for testing
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False  # False
    env_cfg.domain_rand.randomize_friction = False  # False
    env_cfg.domain_rand.push_robots = False  # False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.terrain.mesh_type = "plane"

    env, _ = task_registry.make_env(name=args["task"], args=env_args, env_cfg=env_cfg)
    # ==================================================================

    # =========================================================
    # prepare eat
    print("loading pre_record stds,means...")
    model_path = os.path.join("./Integration_EAT/EAT_runs/", model_name)

    state_mean, state_std = np.load(
        os.path.join(model_path, "model.state_mean.npy")
    ), np.load(os.path.join(model_path, "model.state_std.npy"))

    args["state_mean"] = torch.from_numpy(state_mean).to(device)
    args["state_std"] = torch.from_numpy(state_std).to(device)

    print("loading model...")
    model = LeggedTransformerPro(#这里不能用LeggedTransformerBody
        body_dim=body_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(model_path, "model_best.pt"), map_location=args["device"]
        )
    )
    model.eval()
    # =========================================================

    # =========================================================
    # prepare ippo & loop
    failed_set = set()
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "IPPO_Models"
    for i in range(12):
        train_cfg.runner.checkpoint = i
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=args["task"],
            args=env_args,
            train_cfg=train_cfg,
            log_root="./Integration_EAT",
        )
        policy = ppo_runner.get_inference_policy(device=device)

        failed_set.add(
            do_collect(args=args, env=env, ippo=policy, eat_model=model, fault_tag=[i])
        )
    # do_collect(args=args, env=env, ippo=None, eat_model=model, fault_tag=[11])
    print(failed_set) 
    # =========================================================

    #
