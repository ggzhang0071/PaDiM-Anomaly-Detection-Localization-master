img="nvcr.io/nvidia/pytorch:21.02-py3" 
# img="padim:0.1"

docker run --rm  --gpus all --privileged=true  --workdir /git --name "padim_new"  -e DISPLAY --ipc=host -d --rm  -p 5533:8889  \
-v /home/ubt/she_zhen_code/PaDiM-Anomaly-Detection-Localization-master:/git/PaDiM/ \
-v /home/ubt/she_zhen_code/datasets:/git/datasets \
$img sleep infinity


docker exec -it padim_new /bin/bash
