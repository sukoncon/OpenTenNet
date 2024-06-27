export nnodes=1
export ntasks_per_node=1
export nodes_per_task=1
# export time=$(date +%m-%d-%H:%M:%S)

srun -p llm0_t --quotatype=spot --cpus-per-task=1 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
-x SH-IDC1-10-140-0-142 \
python concate.py \
--use_int8 1