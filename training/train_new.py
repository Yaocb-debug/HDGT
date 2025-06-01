import os
os.environ["NCCL_P2P_DISABLE"] = "1"
import random
from typing import DefaultDict
import warnings
warnings.filterwarnings("ignore")
import gc
import numpy as np
import argparse
import time
import sys
import pickle
import shutil
import importlib
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from waymo_data.collate_func import *
import io
#from metricss_soft_map import soft_map
import scipy.special
import scipy.interpolate as interp
from waymo_dataset import *

parser = argparse.ArgumentParser('Interface for HDGT Training')
##### Optimizer - Scheduler
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--warmup', type=float, default=1.0, help='the number of epoch for warmup')
parser.add_argument('--lr_decay_epoch', type=str, default="4-8-16-24-26", help='the index of epoch where the lr decays to lr*0.5')
parser.add_argument('--num_prediction', type=int,default=6, help='the number of modality')
parser.add_argument('--cls_weight', type=float,default=0.1, help='the weight of classification loss')
parser.add_argument('--reg_weight', type=float,default=50.0, help='the weight of regression loss')

#### Speed Up
parser.add_argument('--num_of_gnn_layer', type=int, default=3, help='the number of HDGT layer')
parser.add_argument('--hidden_dim', type=int, default=256, help='init hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='the dimension of attention head')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--num_worker', type=int, default=8, help='number of worker per dataloader')

#### Setting
parser.add_argument('--agent_drop', type=float, default='0.0', help='the ratio of randomly dropping agent')
parser.add_argument('--data_folder', type=str,default="hdgt_waymo", help='training set')

parser.add_argument('--refine_num', type=int, default=5, help='temporally refine the trajectory')
parser.add_argument('--output_vel', type=str, default="True", help='output in form of velocity') 
parser.add_argument('--cumsum_vel', type=str, default="True", help='cumulate velocity for reg loss')

#### Initialize
parser.add_argument('--checkpoint', type=str, default="none", help='load checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='the index of start epoch (for resume training)')
parser.add_argument('--dev_mode', type=str, default="False", help='develop_mode')

parser.add_argument('--ddp_mode', type=str, default="False", help='False, True, multi_node')
parser.add_argument('--port', type=str, default="31243", help='DDP')

parser.add_argument('--amp', type=str, default="none", help='type of fp16')

#### Log
parser.add_argument('--val_every_train_step', type=int, default=-1, help='every number of training step to conduct one evaluation')
parser.add_argument('--name', type=str, default="hdgt_waymo_dev", help='the name of this setting')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')

args = parser.parse_args()
os.environ["DGLBACKEND"] = "pytorch"



class Logger():
    def __init__(self, lognames):
        self.terminal = sys.stdout
        self.logs = []
        for log_name in lognames:
            self.logs.append(open(log_name, 'w'))
    def write(self, message):
        self.terminal.write(message)
        for log in self.logs:
            log.write(message)
            log.flush()
    def flush(self):
        pass


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()
    ###Distributed
    gpu_count = torch.cuda.device_count()
    global_seed = int(args.port) ## Import!!!! for coherent data splitting across process
    if gpu_count > 1:
        if args.ddp_mode == "multi_node":
            main_worker(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), global_seed, args)
        else:
            mp.spawn(main_worker, nprocs=gpu_count, args=(gpu_count, global_seed, args))
    else:
        main_worker(0, gpu_count, global_seed, args)

## Running for each GPU
def main_worker(gpu, gpu_count, global_seed, args):
    if args.ddp_mode == "multi_node":
        global_rank = int(os.environ["RANK"])
        init_port = "tcp://"+os.environ["MASTER_ADDR"]+":"+os.environ["MASTER_PORT"] 
    else:
        global_rank = gpu
        init_port = "tcp://127.0.0.1:"+args.port
    print(f"Use GPU: {gpu} for training. Global Rank:{global_rank} Global World Size:{gpu_count} Init Port {init_port}")
    print("Process Id:", os.getpid())
    seed_num = random.randint(0, 1000000)
    torch.manual_seed(seed_num+global_rank)
    random.seed(seed_num+global_rank)
    np.random.seed(seed_num+global_rank)

    if gpu_count > 1:
        dist.init_process_group(backend="nccl", world_size=gpu_count, init_method=init_port, rank=global_rank)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:"+str(gpu))

    
    snapshot_dir = None
    if global_rank == 0 and not args.debug:
        setting_name = args.name
        log_dir = "logs/" + str(setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
        
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        sys.stdout = Logger([f"{setting_name}.log", os.path.join(log_dir, f"{setting_name}.log")])
        snapshot_dir = os.path.join(log_dir, "snapshot")
        if not os.path.isdir(snapshot_dir):
            os.makedirs(snapshot_dir)
        print("Log Directory:", os.path.join(log_dir, sys.argv[0]))
        print(args)
        shutil.copyfile(__file__, os.path.join(log_dir, "train_new.py"))
        shutil.copyfile("model_new.py", os.path.join(log_dir, "model_new.py"))

    # 加载数据集
    print("Start Load Dataset")
    train_dataloader, val_dataloader, train_sample_num, val_sample_num = obtain_dataset(global_rank, gpu_count, global_seed, args)

    # 动态获取 time_steps 和 feature_dim
    sample_data = next(iter(train_dataloader))
    a_n_fea = sample_data["graph_lis"].ndata["a_n_fea"]["agent"]
    args.time_steps = a_n_fea.shape[1]
    args.feature_dim = a_n_fea.shape[2]
    print(f"Dataset Dimensions: time_steps={args.time_steps}, feature_dim={args.feature_dim}")

    # 模型初始化
    model_module = importlib.import_module("model_new")
    model = model_module.HDGT_model(input_dim=args.feature_dim, args=args)
    model.apply(model_module.weights_init)
    model = model.to(device, non_blocking=True)


    checkpoint = None
    if args.checkpoint != "none":
        print("Load:", args.checkpoint, gpu)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    step_per_epoch = (train_sample_num + args.batch_size * gpu_count - 1) // (args.batch_size * gpu_count)
    epoch = args.n_epoch

    warmup = args.warmup
    if args.start_epoch > 1:
        warmup = 0.0

    scheduler = model_module.WarmupLinearSchedule(optimizer, step_per_epoch*warmup, step_per_epoch*epoch)
    if args.checkpoint != "none":
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["betas"] = (0.9, 0.95)
        print("lr")
    
    if gpu_count > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    reg_criterion = torch.nn.SmoothL1Loss(reduction="none").to(device)
    if gpu == 0:
        print("start")
    
    if args.amp == "fp16":
        amp_data_type = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print("Use AMP with Type:", amp_data_type)
    elif args.amp == "bf16":
        amp_data_type = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler()
        print("Use AMP with Type:", amp_data_type)
    else:
        amp_data_type = torch.float32 ## Not enabled
        scaler = None

    for epoch in range(args.start_epoch, args.n_epoch+1):
        run_model(dataloader=train_dataloader, num_sample=train_sample_num, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, gpu=gpu, global_rank=global_rank, gpu_count=gpu_count, is_train=True, args=args, val_dataloader=val_dataloader, val_sample_num=val_sample_num, snapshot_dir=snapshot_dir, scaler=scaler, amp_data_type=amp_data_type)
    

def run_model(dataloader, num_sample, model, optimizer, scheduler, epoch, gpu, global_rank, gpu_count, is_train, args, val_dataloader=None, val_sample_num=None, snapshot_dir=None, scaler=None, amp_data_type=None):
    agent_type_lis = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    edge_type_lis = ["self", "other", "a2l", "l2a", "g2a"]
    recorder = {f"{agent_type}_traj_mse": AverageMeter() for agent_type in agent_type_lis}
    recorder.update({f"{edge_type}_edge_mse": AverageMeter() for edge_type in edge_type_lis})
    recorder["loss"] = AverageMeter()
    recorder["traj_loss"] = AverageMeter()
    recorder["edge_loss"] = AverageMeter()

    start_time = time.time()
    if is_train:
        model.train()
        batch_size = args.batch_size
    else:
        model.eval()
        batch_size = args.val_batch_size
        gpu_count = 1

    is_dev = args.dev_mode
    print_freq = 1 if is_dev else 100
    val_every_train_step = 1 if is_dev else args.val_every_train_step
    device = torch.device("cuda:" + str(gpu))
    use_amp = (scaler is not None)

    # 学习率衰减
    lr_decay_epoch = [int(e) for e in args.lr_decay_epoch.split("-")]
    decay_coefficient = 1.0
    for decay_epoch in lr_decay_epoch:
        if epoch >= decay_epoch:
            decay_coefficient *= 0.5
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * decay_coefficient

    with torch.set_grad_enabled(is_train):
        for batch_index, data in enumerate(dataloader, 0):
            data["is_train"] = is_train
            data["gpu"] = gpu
            if "args" in data and global_rank == 0:
                print(f"Warning: data contains args with n_epoch={data['args'].n_epoch}, using args.n_epoch={args.n_epoch}")
            for tensor_name in data["cuda_tensor_lis"]:
                data[tensor_name] = data[tensor_name].to(device, non_blocking=True)
            optimizer.zero_grad()
            num_of_sample = data["graph_lis"].ndata["a_n_fea"]["agent"].shape[0]  # 替换 pred_num_lis

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_data_type):
                recon_traj, recon_edges, agent_traj_mask, edge_masks, original_a_n_fea, edge_hidden_original = model(data, epoch=epoch, n_epoch=args.n_epoch)

                # 计算损失
                traj_loss = F.mse_loss(recon_traj[agent_traj_mask], original_a_n_fea[agent_traj_mask])
                edge_loss = 0.0
                edge_loss_cnt = 0
                for etype in edge_type_lis:
                    if edge_masks[etype].sum() > 0:
                        recon = recon_edges[etype][edge_masks[etype]]
                        original = edge_hidden_original[etype][edge_masks[etype]]
                        edge_loss += F.mse_loss(recon, original)
                        edge_loss_cnt += edge_masks[etype].sum()
                edge_loss = edge_loss / max(edge_loss_cnt, 1)
                loss = traj_loss + edge_loss

            if is_train:
                if loss != 0:
                    if torch.isnan(loss):
                        print(f"Bad Gradients! Epoch {epoch}, Batch {batch_index}")
                        optimizer.zero_grad()
                        del data, recon_traj, recon_edges, agent_traj_mask, edge_masks, original_a_n_fea, edge_hidden_original, loss
                        continue
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                    scheduler.step()

            if global_rank == 0:
                with torch.no_grad():
                    if loss != 0:
                        recorder["loss"].update(loss.item(), num_of_sample)
                        recorder["traj_loss"].update(traj_loss.item(), num_of_sample)
                        recorder["edge_loss"].update(edge_loss.item(), num_of_sample)

                        # 按 Agent 类型计算 MSE
                        a_n_type = data["graph_lis"].ndata["a_n_type"]["agent"]
                        for type_idx, agent_type in enumerate(agent_type_lis):
                            type_mask = a_n_type == type_idx
                            if type_mask.sum() > 0 and agent_traj_mask[type_mask].sum() > 0:
                                mse = F.mse_loss(
                                    recon_traj[type_mask][agent_traj_mask[type_mask]],
                                    original_a_n_fea[type_mask][agent_traj_mask[type_mask]]
                                )
                                recorder[f"{agent_type}_traj_mse"].update(mse.item(), type_mask.sum())

                        # 按边类型计算 MSE
                        for etype in edge_type_lis:
                            if edge_masks[etype].sum() > 0:
                                mse = F.mse_loss(
                                    recon_edges[etype][edge_masks[etype]],
                                    edge_hidden_original[etype][edge_masks[etype]]
                                )
                                recorder[f"{etype}_edge_mse"].update(mse.item(), edge_masks[etype].sum())

                if is_train and ((batch_index + 1) % print_freq) == 0:
                    print_text = f'Epoch: [{epoch}][{(batch_index+1)*batch_size*gpu_count}/{num_sample}-Batch {batch_index}], '
                    print_text += f"Loss {recorder['loss'].avg:.8f}, "
                    print_text += f"Traj Loss {recorder['traj_loss'].avg:.8f}, "
                    print_text += f"Edge Loss {recorder['edge_loss'].avg:.8f}, "
                    for agent_type in agent_type_lis:
                        print_text += f"{agent_type}_traj_mse {recorder[f'{agent_type}_traj_mse'].avg:.4f}, "
                    for etype in edge_type_lis:
                        print_text += f"{etype}_edge_mse {recorder[f'{etype}_edge_mse'].avg:.4f}, "
                    print_text += f"Time(s): {time.time()-start_time:.4f}, "
                    print_text += f"LR: {scheduler.get_last_lr()[0]:.4e}"
                    print(print_text, flush=True)

                    # 重置记录器
                    recorder = {f"{at}_traj_mse": AverageMeter() for at in agent_type_lis}
                    recorder.update({f"{et}_edge_mse": AverageMeter() for et in edge_type_lis})
                    recorder["loss"] = AverageMeter()
                    recorder["traj_loss"] = AverageMeter()
                    recorder["edge_loss"] = AverageMeter()

                    if is_train and ((batch_index + 1) % val_every_train_step == 0 or (val_every_train_step <= 0 and batch_index == len(dataloader) - 1)) and not is_dev:
                        val_model = model.module if gpu_count > 1 else model
                        val_model.eval()
                        run_model(
                            val_dataloader, val_sample_num, val_model, optimizer, scheduler, epoch, gpu, global_rank, 1,
                            is_train=False, args=args, scaler=scaler, amp_data_type=amp_data_type
                        )
                        model.train()
                        file_path = os.path.join(snapshot_dir, f"Epoch_{epoch}_batch{batch_index}.pt")
                        checkpoint = {
                            "model_state_dict": model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "time_steps": args.time_steps,
                            "feature_dim": args.feature_dim
                        }
                        torch.save(checkpoint, file_path)
                        print(f"Epoch {epoch} Batch {batch_index} Save Model")
            del data, recon_traj, recon_edges, agent_traj_mask, edge_masks, original_a_n_fea, edge_hidden_original

    if not is_train:
        print_text = f'****Val Epoch: [{epoch}][{(batch_index+1)*batch_size*gpu_count}/{num_sample}], '
        print_text += f"Loss {recorder['loss'].avg:.8f}, "
        print_text += f"Traj Loss {recorder['traj_loss'].avg:.8f}, "
        print_text += f"Edge Loss {recorder['edge_loss'].avg:.8f}, "
        for agent_type in agent_type_lis:
            print_text += f"{agent_type}_traj_mse {recorder[f'{agent_type}_traj_mse'].avg:.4f}, "
        for etype in edge_type_lis:
            print_text += f"{etype}_edge_mse {recorder[f'{etype}_edge_mse'].avg:.4f}, "
        print_text += f"Time(s): {time.time()-start_time:.4f}"
        print(print_text, flush=True)

if __name__ == '__main__':
    main()