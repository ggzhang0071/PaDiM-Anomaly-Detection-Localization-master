# Padim 用于异常检测
算法流程：
1、 算法做异常检测， 运行：

python3  main.py

# 数据处理获取用于 图像分类
rm -rf  kangqiang_result/croped_images/*
rm  -rf assets_new_new/data/2021-03-05/json_for_classification/*.txt

python generate_json_for_classification.py  2>&1 |tee Logs/generate_json_for_classification_$timestamp.log

python image_crop_and_save_image_label_for_classification_parallel.py  2>&1 |tee Logs/image_crop_and_save_image_label_for_classification_parallel_$timestamp.log

python3 /git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_sel
f-supervised_learning.py

# 获取原始标注和padim segment 共同的扣取

/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/run_dataset_prepare_for_comparsion.sh

# 然后运行  image_classification_model_training-master 
python3 /git/PaDiM-master/image_classification_model_training-master/


