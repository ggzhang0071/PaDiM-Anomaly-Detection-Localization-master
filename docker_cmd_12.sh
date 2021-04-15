img="taikiinoue45/mvtec:mvtec"
#img="padim:v1"
nvidia-docker run --privileged=true  --workdir /git --name "padim"  -e DISPLAY --ipc=host -d --rm  -p 6033:22    \
-v /mnt/di/:/git/dataSet \
-v /disk/zhanggege/nfs_12/PaDiM/:/git/PaDiM/ \
-v /disk/zhanggege/nfs_12/PaDiM-Anomaly-Detection-Localization-master/:/git/PaDiM-master \
-v /disk/zhanggege/nfs_12/results:/git/results \
-v /disk/zhanggege/nfs_12/datasets/:/git/datasets \
$img sleep infinity




