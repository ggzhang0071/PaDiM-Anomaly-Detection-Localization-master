# data preparation with croped images from padim segmentation for image classification 


timestamp=`date +%Y%m%d%H%M%S`
#rm Logs/*.log 

rm  -rf /git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning/*.txt

rm  -rf /git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation_for_classification

rm  -rf  /git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_preparison_for_annotation_and_auto_croped_images_comparison/*.txt

#python3  get_classification_file_with_original_annotation.py

choose_labels="0 3 9"

image_file_path="/git/PaDiM-master/kangqiang_result/croped_images_part_with_classification" 
save_txt_folder="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning"



# run data prepare for semi-supervised learning
python3   get_train_val_test_dataset_from_the_classification_files.py --choose_labels $choose_labels  --image_file_path $image_file_path 
2>&1 |tee Logs/$timestamp.log

original_annotation_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification"
croped_image_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning"
image_data_root="/git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation/image/**/*.jpg"

# image path where the classification results is saved
save_image_path="/git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation_for_classification"
save_txt_folder="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_preparison_for_annotation_and_auto_croped_images_comparison"
    

timestamp=`date +%Y%m%d%H%M%S`
python3 -m pdb get_dataset_for_image_classification_comparion.py --choose_labels $choose_labels --original_annotation_path  $original_annotation_path --image_file_path $--image_file_path --image_data_root  $image_data_root  --save_image_path  $save_image_path --croped_image_path $croped_image_path --save_txt_folder  $save_txt_folder  2>&1 |tee Logs/$timestamp.log


