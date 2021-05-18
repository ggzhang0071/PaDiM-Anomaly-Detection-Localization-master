#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log
rm kangqiang_result/pictures_wide_resnet50_2/*
rm -rf kangqiang_result/segment_image_result_wide_resnet50_2/image/* 
rm -rf kangqiang_result/segment_image_result_wide_resnet50_2/Padim_results_image_label_for_classification.txt 

threshold_coefficient=0.8
for product_id in  "0420QFN-5X6-8L" "0708DFN-8L"  "0713DFN-2X3-8L"  "1101QFN-40L" "1129QFN-4X4-24L" "DFN-5X6-8L" "DFN-5X6-T-8L" 
do
  CUDA_VISIBLE_DEVICES=1,3  python3   main.py  --train_num_samples  0  --test_num_samples 0  --product_class $product_id --threshold_coefficient $threshold_coefficient --save_path /git/PaDiM-master/kangqiang_result 2>&1 |tee Logs/$timestamp.log
done


<<COMMENT

for i in range{1..10}:
do
for product_id in 'QFN-3X3-16L'  '1101QFN-40L' '0713DFN-2X3-8L' 
do
    CUDA_VISIBLE_DEVICES=1,3  python3 main.py --train_num_samples  1500  --test_num_samples 1500 --product_class $product_id  2>&1 |tee Logs/$timestamp.log
done
done
COMMENT
