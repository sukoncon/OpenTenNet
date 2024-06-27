import os
import torch
import argparse
import torch.distributed as dist

from helper import setup_distributed, _nnTo1n, _pairwiseTo1n




if __name__ == "__main__":
    os.environ['PYTHONPROFILER_RECORD'] = '0'
    # 读取参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_select", type=int, default=0, help="0: _nnTo1n, 1: _pairwiseTo1n")
    parser.add_argument("--warmup", type=int, default=1, help="whether to warm up, 0: False, 1: True")
    args = parser.parse_args()

    setup_distributed()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    
    # 确保创建trace_job文件夹
    job_names = ["_nnTo1n", "_pairwiseTo1n"]
    path = "./trace_job{}_{}nodes_{}gpu_per_node/{}".format(\
        job_names[args.job_select], int(os.environ["nnodes"]), torch.cuda.device_count(), os.environ['time'])

    if rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 定义并把tensor放置到单独的GPU上
    device = torch.device("cuda", local_rank)
    tensor_warm = (local_rank * torch.ones((1), dtype=torch.int8)).to(device)
    tensor = (rank * torch.ones((400380, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), dtype=torch.cfloat)).to(device)
        
    job_select = [ _nnTo1n, _pairwiseTo1n]
    job = job_select[args.job_select]()
    job.make_group()
    
    # warm up gpu 
    for i in range(100):
        tensor + tensor * tensor 
    dist.barrier()   

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
            ) as prof:
        if args.warmup:
            job.warmup(tensor_warm, rank)

        tensor_real = torch.view_as_real(tensor)
        job.cal(tensor_real, rank)
        tensor = torch.view_as_complex(tensor_real)

        prof.step()

    prof.export_chrome_trace("{}/rank{}.json".format(path, rank))
    
    if (rank == 0):
        print(f"group list {job.group_lists}")

    if (int(rank/torch.cuda.device_count()) == 0):
        print("\n*********after reduce: {}(rank={})***************".format(tensor[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rank))

'''
*** run: ****
srun -p p4_test --ntasks=4 --ntasks-per-node=2 --gres=gpu:2 python test.py  > test.log 2>&1
or
python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.105.10.10" --master_port=2234 test.py --job_select 5 --warmup 1
'''
