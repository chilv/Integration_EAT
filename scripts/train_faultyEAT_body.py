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
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import D4RLTrajectoryDataset,D4RLTrajectoryDatasetForTert, evaluate_on_env_batch_body, get_dataset_config, partial_traj #, get_d4rl_normalized_score,
from model import LeggedTransformerBody, LeggedTransformerPro, LeggedTransformerBody_Global_Steps
import wandb
from legged_gym.utils import task_registry, Logger 
from tqdm import trange, tqdm
import yaml


def train(args):

    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args["body_dim"]
    ##-----

    # max_eval_ep_len = 500  # max len of one episode
    max_eval_ep_len = args["max_eval_ep_len"]  # max len of one episode
    num_eval_ep = args["num_eval_ep"]          # num of evaluation episodes

    batch_size = args["batch_size"]            # training batch size
    lr = args["lr"]                            # learning rate
    wt_decay = args["wt_decay"]                # weight decay
    warmup_steps = args["warmup_steps"]        # warmup steps for lr scheduler

    context_len = args["context_len"]      # K in decision transformer
    n_blocks = args["n_blocks"]            # num of transformer blocks
    embed_dim = args["embed_dim"]          # embedding (hidden) dim of transformer
    n_heads = args["n_heads"]              # num of transformer heads
    dropout_p = args["dropout_p"]          # dropout probability
    position_encoding_length = args.get("position_encoding_length", 4096)  # length of position encoding

    args["wandboff"] = True if "test" in args["dataset"] else args["wandboff"]

    datafile, i_magic_list, eval_body_vec, eval_env = get_dataset_config(args["dataset"])
    
    file_list = [os.path.join(datafile, f"{i_magic}.pkl") for i_magic in i_magic_list]
    dataset_path_list_raw = [os.path.join(args["dataset_dir"], d) for d in file_list]
    dataset_path_list = []
    for p in dataset_path_list_raw:
        if os.path.isfile(p):
            dataset_path_list.append(p)
        else:
            print(p, " doesn't exist~")

    env_args = get_args()
    env_args.sim_device = args["device"]
    env_args.task = args["task"]
    env_cfg, _ = task_registry.get_cfgs(name =env_args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args["num_eval_ep"])
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.0,1]
    env_cfg.commands.ranges.lin_vel_y = [-0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5,0.5]

    # env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]
    
    env, _ = task_registry.make_env(name = env_args.task, args = env_args, env_cfg = env_cfg)
    
    # saves model and csv in this directory
    log_dir = args["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args["device"])
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # note_abbr = ''.join([word[0] if not word.isnumeric() else word for word in np.concatenate([x.split() for x in re.split('(\d+)',args.note)]) ]).upper()
    run_idx = 0
    algo = args["algorithm_name"]
    model_file_name = f"{algo}_" + args["dataset"].split('.')[0].upper().replace("_", "").replace("-", "") + "_" + str(run_idx).zfill(2)
    previously_saved_model = os.path.join(log_dir, model_file_name)
    while os.path.exists(previously_saved_model):
        run_idx += 1
        model_file_name = f"{algo}_" + args["dataset"].split('.')[0].upper().replace("_", "").replace("-", "") + "_" + str(run_idx).zfill(2)
        previously_saved_model = os.path.join(log_dir, model_file_name)

    os.makedirs(os.path.join(log_dir, model_file_name))
    save_model_path = os.path.join(log_dir, model_file_name, "model")

    with open(os.path.join(log_dir, model_file_name, "note.txt"), 'w') as f:
        f.write(args["note"])
        
    log_csv_path = os.path.join(log_dir, model_file_name, "log.csv")

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "body_loss", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + str(dataset_path_list))
    print("model save path: " + save_model_path+".pt (.jit)")
    print("log csv save path: " + log_csv_path)

    args["dataset path"] = str(dataset_path_list)
    args["model save path"] = save_model_path+".pt"
    args["log csv save path"] = log_csv_path
    # pdb.set_trace()
    if not args["wandboff"]:
        try:
            wandb.init(config=args, project="my_EAT_test", name=model_file_name, mode="online", notes=args["note"])
        except:
            print("wandb disabled")
            wandb.init(config=args, project="my_EAT_test", name=model_file_name, mode="disabled", notes=args["note"])

    #---------------------------------------------------------------------------------------------------------------
    print("Loding paths for each robot model...")
    #加载轨迹部分
    big_list = []
    for pkl in tqdm(dataset_path_list):  
        with open(pkl, 'rb') as f:
            thelist = pickle.load(f)

        assert "bodies" in thelist[0] or "bodys" in thelist[0] or "body" in thelist[0]
        if args["cut"] == 0:
            big_list = big_list + thelist
        else:
            big_list = big_list + thelist[:args['cut']]

    # body_preds = None
    with open(os.path.join(log_dir, model_file_name,"args.yaml") , "w") as log_for_arg:
        print(yaml.dump(args, log_for_arg))
        
    with open(os.path.join(log_dir, model_file_name,"env_args.yaml") , "w") as log_for_arg:
        print(yaml.dump(env_args, log_for_arg))
        
    traj_dataset = D4RLTrajectoryDataset(big_list, context_len, leg_trans_pro=True)
    assert body_dim == traj_dataset.body_dim
    
    state_mean, state_std = traj_dataset.get_state_stats(body=False)
    
    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    print("DataLoader complete!")
    n_epochs = int(1e6 / len(traj_data_loader) * args["n_epochs_ref"])  #default args["n_epochs_ref"] = 1
    num_updates_per_iter = len(traj_data_loader)   
    
    np.save(f"{save_model_path}.state_mean", state_mean)
    np.save(f"{save_model_path}.state_std", state_std)
    #---------------------------------------------------------------------------------------------------------------------------
    # if slices == 0:
    print("model preparing")
    # if args.get("Global_Step_Embeddings", False): # Not Work
    #     model = LeggedTransformerBody_Global_Steps(
    #             body_dim=body_dim,
    #             state_dim=state_dim,
    #             act_dim=act_dim,
    #             n_blocks=n_blocks,
    #             h_dim=embed_dim,
    #             context_len=context_len,
    #             n_heads=n_heads,
    #             drop_p=dropout_p,
    #             state_mean=state_mean,
    #             state_std=state_std,
    #         ).to(device)
    # else:
    # Local_Context_Len Positional Encoding 
    model = LeggedTransformerBody(
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
            position_encoding_length=position_encoding_length,
        ).to(device)
    
    if args["model_dir"] is not None:
        model.load_state_dict(torch.load(
            os.path.join( args["model_dir"], "model_best.pt" ), map_location = args["device"]
        ))
        
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

    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    pred_body = args["pred_body"]
    max_d4rl_score = -1.0
    total_updates = 0

    # global_train_step = 0
    inner_bar = tqdm(range(num_updates_per_iter), leave = False)
    state_mean = model.state_mean.to(device)
    state_std = model.state_std.to(device)
        
    for epoch in trange(n_epochs):
        # pdb.set_trace()
        log_action_losses = []
        log_body_losses = []
        model.train()

        for states, actions, bodies, gt_bodies, traj_mask in iter(traj_data_loader):

            # timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim

            actions = actions.to(device)        # B x T x act_dim
            bodies = bodies.to(device)  # B x T x body_dim
            B, T, D = bodies.shape
            gt_bodies = gt_bodies.to(device)  # B x T x body_dim
            body_target = torch.clone(gt_bodies).detach().to(device)
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            if pred_body:
                # body_1 = torch.ones_like(actions)   #body full of 1.0
                
                _, _, body_preds = model.forward(
                                                                states=states,
                                                                actions=actions,
                                                                bodies = bodies
                                                                # body=body
                                                            )
                
                # mesure body loss
                # 预测的body是当前body
                body_preds = body_preds.clamp(0.0, 1.0)
                # only consider non padded elements
                target_joints = torch.argmin(gt_bodies, dim=-1).unsqueeze(2)

                #仅保留主要loss
                body_main_loss = F.mse_loss(
                    body_preds.gather(
                        dim=2,index = target_joints), 
                    body_target.gather(
                        dim=2,index = target_joints), 
                    reduction='mean')
                
                #adversary loss
                _, idx1 = torch.sort(torch.abs(body_preds.detach()-body_target), dim=-1, descending=False)
                p = (idx1!=target_joints.expand(-1,-1,12)).nonzero()[:,-1].reshape((B,T,D-1)) #非最小元素的所有坐标
                p = p[:,:,:2]
                #按理说应该要body_adversary尽可能远离body_target
                # 但由于目前的setting主要是从1下降到任意坏损值，所以偷懒写成body_adversary离1的距离越近越好
                body_adversary_loss = torch.exp(
                    (body_preds.gather(dim=2,index = p) 
                    - body_target.gather(dim=2,index = target_joints).expand(-1,-1,2)).mean()
                )
                
                body_preds = body_preds.view(-1, body_dim)[traj_mask.view(-1,) > 0]
                body_target = body_target.view(-1, body_dim)[traj_mask.view(-1,) > 0]
                body_loss = body_main_loss + body_adversary_loss 
                + F.mse_loss(body_preds, body_target, reduction='mean')
            
            #measure action loss
            _, action_preds, _ = model.forward(
                                                            states=states,
                                                            actions=actions,
                                                            # body = body_1
                                                            bodies=bodies
                                                        )
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            total_loss = args["action_loss_w"] * action_loss
            if pred_body:
                total_loss += (args["body_loss_w"] * body_loss)
            total_loss = total_loss.to(torch.float)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            if pred_body:
                log_body_losses.append(body_loss.detach().cpu().item())
            log_action_losses.append(action_loss.detach().cpu().item())

            if not args["wandboff"]:
                if pred_body:
                    wandb.log({"Loss_body": body_loss.detach().cpu().item(), 
                           "Loss_body_main": body_main_loss.detach().cpu().item(),
                           "Loss_body_adv": body_adversary_loss.detach().cpu().item(),
                           "Loss_action": action_loss.detach().cpu().item(),
                           })
                else:
                    wandb.log({"Loss_action": action_loss.detach().cpu().item()})

            total_updates += num_updates_per_iter

            inner_bar.update(1)
        inner_bar.reset()
        #end one epoch ^
        #=====================================================================================================

        if epoch % max(int(5000/num_updates_per_iter), 1) == 0:
            # evaluate action accuracy
            results = evaluate_on_env_batch_body(model = model, device=device, context_len=context_len, env=env, 
                                                body_target=eval_body_vec, num_eval_ep = num_eval_ep, 
                                                max_test_ep_len = max_eval_ep_len, state_mean = state_mean, 
                                                state_std = state_std, body_pre = pred_body)

        
            eval_avg_reward = results['eval/avg_reward']
            eval_avg_ep_len = results['eval/avg_ep_len']
            
            mean_body_loss = np.mean(log_body_losses)
            mean_action_loss = np.mean(log_action_losses)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

            log_str = ("=" * 60 + '\n' +
                    "time elapsed: " + time_elapsed  + '\n' +
                    "num of updates: " + str(total_updates) + '\n' +
                    "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                    "body loss: " +  format(mean_body_loss, ".5f") + '\n' +
                    "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                    "eval avg ep len: " + format(eval_avg_ep_len, ".5f") #+ 
                )

            tqdm.write(log_str)
            if not args["wandboff"]:
                wandb.log({"Evaluation Score": eval_avg_reward, "Episode Length": eval_avg_ep_len, "Total Steps": total_updates})
            
            log_data = [time_elapsed, total_updates, mean_body_loss, mean_action_loss,
                        eval_avg_reward, eval_avg_ep_len]

            csv_writer.writerow(log_data)
            
            # save model
            if eval_avg_reward >= max_d4rl_score:
                tqdm.write("saving max score model at: " + save_model_path+"_best.pt(.jit)")
                tqdm.write("max score: " + format(max_d4rl_score, ".5f"))
                torch.save(model.state_dict(), save_model_path+"_best.pt")
                traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
                traced_script_module.save(save_model_path+"_best.jit")
                max_d4rl_score = eval_avg_reward
        #end one eval & log
        #========================================================================

        if epoch % 100 == 0:
            torch.save(model.state_dict(), save_model_path+str(epoch)+"epoch.pt")
            traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
            traced_script_module.save(save_model_path+str(epoch)+"epoch.jit")

        # tqdm.write("saving current model at: " + save_model_path+".pt(.jit)") 
        # print("saving current model at: " + save_model_path+".pt(.jit)")
        # if total_updates % 10 == 0:
        torch.save(model.state_dict(), save_model_path+str(epoch%10)+".pt")
        traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
        traced_script_module.save(save_model_path+str(epoch%10)+".jit")
    #end training
    #=======================================================================================

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
    if not args["wandboff"]:
        wandb.finish()

# def preyaml(args):
#     env_args = get_args()
#     with open("/NAS2020/Workspaces/DRLGroup/wuxinyuan/leggedrobots/Integration_EAT/scripts/args.yaml", "w") as fyaml:
#         print(yaml.dump_all([args, env_args], fyaml))
    
#     return

if __name__ == "__main__":

    with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)
    train(args)
    # preyaml(args)
