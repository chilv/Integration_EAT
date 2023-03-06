import argparse
from argparse import Namespace
import os
import pickle
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
import random
import csv
from datetime import datetime
import copy
import math
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import D4RLTrajectoryDataset, evaluate_on_env,  evaluate_on_env_batch_body, get_dataset_config #, get_d4rl_normalized_score,
from model import DecisionTransformer, LeggedTransformer, LeggedTransformerPro, MLPBCModel
import wandb
# from singlea1 import A1
# from a1wrapper import A1
from legged_gym.utils import task_registry, Logger
from tqdm import trange, tqdm

def partial_traj(dataset_path_list, context_len=20, rtg_scale=1000, body_dim=12):
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


def train(args):

    rtg_scale = 1000      # normalize returns to go

    state_dim = 48
    act_dim = 12
    # body_dim = 3
    ##!--一些需要注意的改动
    #为了适应四条腿分别出现故障的情况 需要改动body_dim项
    #原论文为（前腿长，后腿长，去赶场）
    #改为 12个关节每个一维的形式
    body_dim = 12
    ##-----

    # max_eval_ep_len = 500  # max len of one episode
    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = n_epochs x num_updates_per_iter
    # n_epochs = args.n_epochs
    # max_train_steps = args.max_train_steps
    # num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability


    datafile, i_magic_list, eval_body_vec, eval_env = get_dataset_config(args.dataset)
    
    # file_list = [f"a1magic{i_magic}-{datafile}.pkl" for i_magic in i_magic_list]
    file_list = [os.path.join(datafile, f"{i_magic}.pkl") for i_magic in i_magic_list]
    dataset_path_list_raw = [os.path.join(args.dataset_dir, d) for d in file_list]
    dataset_path_list = []
    for p in dataset_path_list_raw:
        if os.path.isfile(p):
            dataset_path_list.append(p)
        else:
            print(p, " doesn't exist~")

    # env = A1(num_envs=args.num_eval_ep, robot=eval_env, noise=args.noise)
    env_args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name =env_args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_eval_ep)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]
    env, _ = task_registry.make_env(name = args.task, args = env_args, env_cfg = env_cfg)
    # env = A1(num_envs=args.num_eval_ep, noise=args.noise)#这里eval_env编译不通过，因为注册表中没有该环境，暂时跳过试一下
    
    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # note_abbr = ''.join([word[0] if not word.isnumeric() else word for word in np.concatenate([x.split() for x in re.split('(\d+)',args.note)]) ]).upper()
    run_idx = 0
    algo = "EAT" if not args.nobody else "NB"
    model_file_name = f"{algo}_" + args.dataset.split('.')[0].upper().replace("_", "").replace("-", "") + "_" + str(run_idx).zfill(2)
    previously_saved_model = os.path.join(log_dir, model_file_name)
    while os.path.exists(previously_saved_model):
        run_idx += 1
        model_file_name = f"{algo}_" + args.dataset.split('.')[0].upper().replace("_", "").replace("-", "") + "_" + str(run_idx).zfill(2)
        previously_saved_model = os.path.join(log_dir, model_file_name)

    os.makedirs(os.path.join(log_dir, model_file_name))
    save_model_path = os.path.join(log_dir, model_file_name, "model")

    with open(os.path.join(log_dir, model_file_name, "note.txt"), 'w') as f:
        f.write(args.note)
        
    log_csv_path = os.path.join(log_dir, model_file_name, "log.csv")

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + str(dataset_path_list))
    print("model save path: " + save_model_path+".pt (.jit)")
    print("log csv save path: " + log_csv_path)


    config=vars(args)
    config["dataset path"] = str(dataset_path_list)
    config["model save path"] = save_model_path+".pt"
    config["log csv save path"] = log_csv_path
    # pdb.set_trace()
    wandb.init(config=config, project="my_EAT_test", name=model_file_name, mode="online" if not args.wandboff else "disabled", notes=args.note)
    
    # turns = math.ceil(len(dataset_path_list)/20.0)  #轨迹太多的时候拆分成为之多20段一组的情况
    # print(f"the training process will be sliced to {turns} parts ... ")
    # for slices in range(turns):
    #---------------------------------------------------------------------------------------------------------------
    print("Loding paths for each robot model...")
    #加载轨迹部分
    if len(dataset_path_list) < 103:
        traj_dataset, state_mean, state_std = partial_traj(dataset_path_list)
    else:   #当轨迹过多时进行拆分
        dataset_list, state_mean_list, state_std_list, xn = [],[],[],[]
        print(f"divide trajs to {math.ceil(len(dataset_path_list)/12.0)} parts")
        for i in range( math.ceil(len(dataset_path_list)/12.0) ):
            print(f"getting {i+1} parts trajs...")
            traj_dataset, state_mean, state_std = partial_traj(dataset_path_list[12 * i : 12 * i + 12])
            dataset_list.append(traj_dataset)
            state_mean_list.append(state_mean)
            state_std_list.append(state_std)
            # body_mean_list.append(body_mean)
            # body_std_list.append(body_std)
            xn.append(len(traj_dataset))
            
        #合并轨迹与各项均值方差
        traj_dataset = torch.utils.data.ConcatDataset(dataset_list)
        state_ndarray = np.expand_dims(xn, 1).repeat(48,1)
        body_ndarray = np.expand_dims(xn, 1).repeat(12,1)
        state_mean = np.sum(state_mean_list*state_ndarray,0)/np.sum(xn)
        state_std = np.sum(state_std**2*state_ndarray,0)/np.sum(xn)  #由于numpy.std默认求有偏样本标准差 所以这里分母是n
        # body_mean = np.sum(body_mean_list*body_ndarray,0)/np.sum(xn)
        # body_std = np.sum(body_std**2*body_ndarray,0)/np.sum(xn)

    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    print("DataLoader complete!")
    n_epochs = int(1e6 / len(traj_data_loader) * args.n_epochs_ref)
    num_updates_per_iter = len(traj_data_loader)
    

    # data_iter = iter(traj_data_loader)

    ## get state stats from dataset
    # state_mean, state_std, body_mean, body_std = traj_dataset.get_state_stats(body=True)    
    
    np.save(f"{save_model_path}.state_mean", state_mean)
    np.save(f"{save_model_path}.state_std", state_std)
    # np.save(f"{save_model_path}.body_mean", body_mean)
    # np.save(f"{save_model_path}.body_std", body_std)
    #---------------------------------------------------------------------------------------------------------------------------
    # if slices == 0:
    print("model preparing")
    if args.nobody:
        model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
                state_mean=state_mean,
                state_std=state_std,
                use_rtg =False,
            ).to(device)
    else:
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
                    # body_mean=body_mean,
                    # body_std=body_std
                ).to(device)
        
    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay,
                        betas=(0.9, 0.95)
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    max_d4rl_score = -1.0
    total_updates = 0

    # global_train_step = 0
    inner_bar = tqdm(range(num_updates_per_iter), leave = False)
    state_mean = model.state_mean.to(device)
    state_std = model.state_std.to(device)
    for epoch in trange(n_epochs):
        # pdb.set_trace()

        log_action_losses = []
        model.train()

        for timesteps, states, actions, body, traj_mask in iter(traj_data_loader):


            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim

            # states = (states - state_mean)/state_std

            actions = actions.to(device)        # B x T x act_dim
            # returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            body = body.to(device).type(actions.dtype) # B x T x body_dim
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            if not args.nobody:
                state_preds, action_preds, return_preds = model.forward(
                                                                timesteps=timesteps,
                                                                states=states,
                                                                actions=actions,
                                                                body=body
                                                            )
            else:
                state_preds, action_preds, return_preds = model.forward(
                                                                timesteps=timesteps,
                                                                states=states,
                                                                actions=actions
                                                            )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            # print("=======================TARGET======================")
            # print(action_target.detach())
            # print("=======================PRED======================")
            # print(action_preds.detach())
            # print("=======================LOSS======================")
            # print(action_loss.detach())

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

            wandb.log({"Loss": action_loss.detach().cpu().item()})

            total_updates += num_updates_per_iter

            inner_bar.update(1)

        
        inner_bar.reset()

        if epoch % max(int(5000/num_updates_per_iter), 1) == 0:
            # evaluate action accuracy

            results = evaluate_on_env_batch_body(model, device, context_len, env, eval_body_vec, 1,
                                                num_eval_ep, max_eval_ep_len, state_mean, state_std, nobody=args.nobody)

        
            eval_avg_reward = results['eval/avg_reward']
            eval_avg_ep_len = results['eval/avg_ep_len']
            eval_d4rl_score = results['eval/avg_reward']
            # eval_avg_reward = eval_avg_ep_len = eval_d4rl_score = 1000
            
            mean_action_loss = np.mean(log_action_losses)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)


            log_str = ("=" * 60 + '\n' +
                    "time elapsed: " + time_elapsed  + '\n' +
                    "num of updates: " + str(total_updates) + '\n' +
                    "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                    "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                    "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + 
                    "eval score: " + format(eval_d4rl_score, ".5f")
                )

            # print(log_str)
            tqdm.write(log_str)
            # print("")

            wandb.log({"Evaluation Score": eval_avg_reward, "Episode Length": eval_avg_ep_len, "D4RL Score": eval_d4rl_score, 
                    "Average Loss": mean_action_loss, "Total Steps": total_updates})
            
            log_data = [time_elapsed, total_updates, mean_action_loss,
                        eval_avg_reward, eval_avg_ep_len,
                        eval_d4rl_score]

            csv_writer.writerow(log_data)

            # save model
            if eval_d4rl_score >= max_d4rl_score:
                tqdm.write("saving max score model at: " + save_model_path+"_best.pt(.jit)")
                tqdm.write("max score: " + format(max_d4rl_score, ".5f"))
                torch.save(model.state_dict(), save_model_path+"_best.pt")
                traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
                traced_script_module.save(save_model_path+"_best.jit")
                max_d4rl_score = eval_d4rl_score

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), save_model_path+str(epoch)+"epoch.pt")
            traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
            traced_script_module.save(save_model_path+str(epoch)+"epoch.jit")

        # tqdm.write("saving current model at: " + save_model_path+".pt(.jit)") 
        # print("saving current model at: " + save_model_path+".pt(.jit)")
        # if total_updates % 10 == 0:
        torch.save(model.state_dict(), save_model_path+str(epoch%10)+".pt")
        traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
        traced_script_module.save(save_model_path+str(epoch%10)+".jit")


    tqdm.write("=" * 60)
    tqdm.write("finished training!")
    tqdm.write("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    tqdm.write("started training at: " + start_time_str)
    tqdm.write("finished training at: " + end_time_str)
    tqdm.write("total training time: " + time_elapsed)
    tqdm.write("max score: " + format(max_d4rl_score, ".5f"))
    tqdm.write("saved max score model at: " + save_model_path+"_best.pt")
    tqdm.write("saved last updated model at: " + save_model_path+".pt")
    tqdm.write("=" * 60)

    wandb.finish()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='flawedppo')

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=1024)      #事实上此参数决定了测试环境数量 原值10
    parser.add_argument('--noise', type=int, help="noisy environemnt for evaluation", default=0)

    parser.add_argument('--dataset_dir', type=str, default='Integration_EAT/data/')
    parser.add_argument('--log_dir', type=str, default='Integration_EAT/EAT_runs/')
    parser.add_argument('--cut', type=int, default=0)

    parser.add_argument('--context_len', type=int, default=20)  #50 试一下
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=128)   #! 试图修改transformer规模
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

    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--seed', type=int, default=66)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # args.wandboff = True    #当无法连接wandb时使用
    train(args)
