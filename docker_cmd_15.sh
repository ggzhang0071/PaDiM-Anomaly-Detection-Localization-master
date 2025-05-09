img="taikiinoue45/mvtec:mvtec"
# img="padim:0.1"

docker run --rm  --gpus all --privileged=true  --workdir /git --name "padim"  -e DISPLAY --ipc=host -d --rm  -p 5513:8889  \
-v /home/ubt/she_zhen_code/PaDiM-Anomaly-Detection-Localization-master:/git/PaDiM/ \
-v /home/ubt/she_zhen_code/datasets:/git/datasets \
$img sleep infinity


docker exec -it padim /bin/bash
