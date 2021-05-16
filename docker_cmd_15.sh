#img="taikiinoue45/mvtec:mvtec"
img="padim:0.1"

nvidia-docker run --privileged=true  --workdir /git --name "padim"  -e DISPLAY --ipc=host -d --rm  -p 5513:8889  \
-v /mnt/di/:/git/dataSet \
-v /disk/zhanggege/moco/:/git/moco/ \
-v /disk/zhanggege/results:/git/results \
-v /disk/zhanggege/datasets/:/git/datasets \
$img sleep infinity
