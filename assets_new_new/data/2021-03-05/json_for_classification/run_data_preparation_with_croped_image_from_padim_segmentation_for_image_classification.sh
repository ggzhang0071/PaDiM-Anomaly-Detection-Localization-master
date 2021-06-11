# data preparation with croped images from padim segmentation for image classification 

timestamp=`date +%Y%m%d%H%M%S`
#rm Logs/*.log 

rm -rf  kangqiang_result/croped_images/*
rm  -rf assets_new_new/data/2021-03-05/json_for_classification/*.txt

python get_classification_file_with_original_annotation.py  2>&1 |tee Logs/$timestamp.log

timestamp=`date +%Y%m%d%H%M%S`
python get_train_val_test_dataset_from_the_classification_files.py  2>&1 |tee Logs/$timestamp.log   
