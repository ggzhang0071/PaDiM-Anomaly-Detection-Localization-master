#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`
save_path=shezhen_result
rm Logs/*.log
rm ${save_path}/pictures_wide_resnet50_2/*
rm -rf ${save_path}/segment_image_result_wide_resnet50_2/image/* 
rm -rf ${save_path}/segment_image_result_wide_resnet50_2/*.txt
# "0420QFN-5X6-8L" "0708DFN-8L"  "0713DFN-2X3-8L"  "1101QFN-40L" "1129QFN-4X4-24L" "DFN-5X6-8L" "DFN-5X6-T-8L" 
# 
threshold_coefficient=0.8

python3 -m pdb main.py  --train_num_samples  0  --test_num_samples 1500  --product_class "she_zhen_data" --threshold_coefficient $threshold_coefficient  --gaussian_blur_kernel_size 5 --save_path /git/PaDiM-master/kangqiang_result 2>&1 |tee Logs/$timestamp.log


<<COMMENT
for product_id in  "0420QFN-5X6-8L" "0708DFN-8L"  "0713DFN-2X3-8L"  "1101QFN-40L" "1129QFN-4X4-24L" "DFN-5X6-8L" "DFN-5X6-T-8L" 
do
  CUDA_VISIBLE_DEVICES=1,3,2,0  python3   main.py  --train_num_samples  0  --test_num_samples 0  --product_class $product_id --threshold_coefficient $threshold_coefficient --gaussian_blur_kernel_size 3 --save_path /git/PaDiM-master/kangqiang_result_3 2>&1 |tee Logs/$timestamp.log
done
COMMENT

<<COMMENT
for product_id in 'QFN-3X3-16L'  '1101QFN-40L' '0713DFN-2X3-8L' 
do
    CUDA_VISIBLE_DEVICES=1,3,2,0  python3 main.py --train_num_samples  1500  --test_num_samples 1500 --product_class $product_id  2>&1 |tee Logs/$timestamp.log
done
COMMENT

<<COMMENT
for product_id in 'QFN-3X3-16L'  '1101QFN-40L' '0713DFN-2X3-8L' 
do
    CUDA_VISIBLE_DEVICES=1,3,2,0  python3 main.py --train_num_samples  0  --test_num_samples 0 --product_class $product_id  2>&1 |tee Logs/$timestamp.log
done
COMMENT



