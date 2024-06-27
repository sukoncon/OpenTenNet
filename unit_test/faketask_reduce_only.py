import torch
import torch.distributed as dist
import re
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
import os
import multiprocessing as mp
from math import ceil
import time
from torch.profiler import profile, record_function, ProfilerActivity
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape
from helper import setup_distributed, _nnTo1n, _pairwiseTo1n


mgmodes = 3
split = 512
torch.random.manual_seed(0)

############# SETTING UP FOR MULTI_NODE CONFIGURATION ############################################################
setup_distributed()
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])   
local_world_size = torch.cuda.device_count()
##################################################################################################################
############# SETTING UP FOR REDUCE CONFIGURATION ################################################################
import argparse
# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument("--job_select", type=int, default=0, help="0: _nnTo1n, 1: _pairwiseTo1n")
parser.add_argument("--warmup", type=int, default=1, help="whether to warm up, 0: False, 1: True")
args = parser.parse_args()


# 确保创建trace_job文件夹
job_names = ["_nnTo1n", "_pairwiseTo1n"]
path = "./trace_fake_reduce_only{}_{}nodes_{}gpu_per_node/{}".format(\
    job_names[args.job_select], int(os.environ["nnodes"]), torch.cuda.device_count(), os.environ['time'])

if rank == 0:
    if not os.path.exists(path):
        os.makedirs(path)

# 定义并把tensor放置到单独的GPU上
device = torch.device("cuda", local_rank)
    
job_select = [ _nnTo1n, _pairwiseTo1n]
job = job_select[args.job_select]()
job.make_group()
if args.warmup:
    tensor_warm = (local_rank * torch.ones((1), dtype=torch.int8)).to(device)
    job.warmup(tensor_warm, rank)
    del tensor_warm
node_index = rank//local_world_size
#################################################################################################################


if local_rank == 0:
    ans = torch.randn(size = [437292, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 1:
    ans = torch.randn([416038, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 2:
    ans = torch.randn([407181, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 3:
    ans = torch.randn([386036, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 4:
    ans = torch.randn([360009, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 5:
    ans = torch.randn([342263, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 6:
    ans = torch.randn([334581, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)
elif local_rank == 7:
    ans = torch.randn([316600, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype = torch.cfloat, device = device)

torch.cuda.empty_cache()
dist.barrier()
ntask = 1
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        ) as prof:
    torch.cuda.synchronize(); time_begin = time.time()    
    tensor_real = torch.view_as_real(ans)
    job.cal(tensor_real, rank)
    ans_node0 = torch.view_as_complex(tensor_real)
    torch.cuda.synchronize(); time_end = time.time()
    prof.step()

if node_index == 0:
    prof.export_chrome_trace("{}/rank{}.json".format(path, rank))
if node_index == 0:
    print(f"reduce on rank {rank} used time {time_end-time_begin}s")

# dist.barrier()
# if node_index == 0:
#     print(f"ans_node0 on rank {rank} is {torch.flatten(ans_node0.real)[0]}(after reduce)")
# dist.barrier()

# if (rank == 0):
#     if isinstance(job.group_lists, list):
#         print(f"job.group_lists {job.group_lists}")
#     else:
#         for i in job.group_lists:
#             print(f"job.group_lists[{i}] {job.group_lists[i]}")


