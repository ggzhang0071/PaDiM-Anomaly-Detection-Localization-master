timestamp=`date +%Y%m%d%H%M%S`
#rm Logs/*.log 

rm -rf  kangqiang_result/croped_images/*
rm  -rf assets_new_new/data/2021-03-05/json_for_classification/*.txt

python generate_json_for_classification.py  2>&1 |tee Logs/generate_json_for_classification_$timestamp.log

python image_crop_and_save_image_label_for_classification_parallel.py  2>&1 |tee Logs/image_crop_and_save_image_label_for_classification_parallel_$timestamp.log


