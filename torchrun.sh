clear
export time=
export nodes_per_task=1
export ntasks_per_node=1

torchrun --nnodes=1 --nproc-per-node=${ntasks_per_node} \
scripts/faketask.py \
--data_type 1 --ntask 8 --tensorNetSize faketask

# --typeCom int4kernel/int8kernel/HalfKernel