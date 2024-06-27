export nnodes=4
export ntasks_per_node=8
export time=$(date +%m-%d-%H:%M:%S)
srun -p p4_test --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} --gres=gpu:${ntasks_per_node} \
 python unit_test.py   --job_select 1 --warmup 1 > \
log/test_${time}.log 2>&1

