export NCCL_DEBUG=INFO
export nnodes=1
export ntasks_per_node=8
export time=$(date +%m-%d-%H:%M:%S)
torchrun --nnodes=1 --nproc-per-node=${ntasks_per_node} \
test.py   --job_select 0 --warmup 1 > nccl.log 2>&1
