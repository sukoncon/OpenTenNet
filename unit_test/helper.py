import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import argparse
import time
import subprocess
import math


class _pairwiseTo1n():
    def __init__(self):
        self.groups = None
        self.nnodes = int(os.environ["nnodes"])
        self.num = int(math.log(self.nnodes, 2))
        self.gpus = torch.cuda.device_count()

    def make_group(self):
        self.groups = {}
        self.group_lists = {}
        pairwise_groups = self.nnodes
        for iter in range(self.num):
            pairwise_groups = int(pairwise_groups/2)
            group_list = []; group = []
            for n in range(pairwise_groups ):
                for gpu in range(self.gpus):
                    group_list.append([n* self.gpus + gpu, (n+pairwise_groups)*self.gpus + gpu])
                    group.append(dist.new_group([n* self.gpus + gpu, (n+pairwise_groups)*self.gpus + gpu]))
            self.group_lists[iter] = group_list
            self.groups[iter] = group

    def warmup(self, tensor_warm, rank, num_loop=1):
        for _ in range(num_loop):
            pairwise_groups = self.nnodes
            for iter in range(self.num):
                if rank < pairwise_groups*self.gpus:
                    dist.reduce(tensor_warm, rank % int(pairwise_groups*self.gpus/2), op=ReduceOp.SUM, \
                    group=self.groups[iter][rank % int(pairwise_groups*self.gpus/2)])
                pairwise_groups = int(pairwise_groups/2)
    
    def cal(self, tensor, rank):
        pairwise_groups = self.nnodes
        for iter in range(self.num):
            if rank < pairwise_groups*self.gpus:
                dist.reduce(tensor, rank % int(pairwise_groups*self.gpus/2), op=ReduceOp.SUM, \
                group=self.groups[iter][rank % int(pairwise_groups*self.gpus/2)])
            pairwise_groups = int(pairwise_groups/2)


class _nnTo1n():
    def __init__(self):
        self.groups = None
        self.nnodes = int(os.environ["nnodes"])

    def make_group(self):
        self.groups = []
        self.group_lists = []
        for i in range(torch.cuda.device_count()):
            self.groups.append(dist.new_group([index for index in range(i, self.nnodes*torch.cuda.device_count(), torch.cuda.device_count())]))
            self.group_lists.append([index for index in range(i, self.nnodes*torch.cuda.device_count(), torch.cuda.device_count())])

    def warmup(self, tensor_warm, rank, num_loop=1):
        for _ in range(num_loop):
            self.cal(tensor_warm, rank)
    
    def cal(self, tensor, rank):
        dist.reduce(tensor, rank%torch.cuda.device_count(), op=ReduceOp.SUM, group=self.groups[rank%torch.cuda.device_count()])

def get_MaterAddr(file_path, timeout = 60):
    start_time = time.time()  
    while not os.path.exists(file_path):  
        if time.time() - start_time > timeout:  
            break
        time.sleep(1) 
    if os.path.isfile(file_path):  
        with open(file_path, 'r') as f:
            os.environ["MASTER_ADDR"] = f.readline().strip()
    else:  
        raise ValueError("%s isn't a file! Please write your master address correctly!" % file_path)

def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()
    
    if "SLURM_JOB_ID" in os.environ:
        # if split the project in multiple srun
        if "num_srun" in os.environ and int(os.environ["num_srun"]) > 1:
            # rank = int(os.environ["SLURM_PROCID"])
            rank = int(os.environ["SLURM_PROCID"]) + int(os.environ["RANK_BASE"])
            world_size = int(os.environ["WORLD_SIZE"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            if "MASTER_ADDR" not in os.environ and os.environ["RANK_BASE"] == "0":
                os.environ["MASTER_ADDR"] = addr
                if rank == 0:
                    # 将MASTER_ADDR写入文件
                    with open(f"log/MASTER_ADDR_{os.environ['time']}.txt", 'w') as f:
                        f.write(addr)
            if "MASTER_ADDR" not in os.environ:
                file_path = f"log/MASTER_ADDR_{os.environ['time']}.txt"
                get_MaterAddr(file_path, timeout = 600)
        else: 
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
        
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        os.environ["nnodes"] = str(int(world_size/torch.cuda.device_count()))

    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ["nnodes"] = str(int(world_size/torch.cuda.device_count()))

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )   
