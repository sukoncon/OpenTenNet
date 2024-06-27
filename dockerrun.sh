sudo docker run --name suzhongling -it \
-v ${PWD}:/OpenTenNet \
-w /OpenTenNet \
--gpus all \
--shm-size=21474836480 \
opentennet:v0 /bin/bash
