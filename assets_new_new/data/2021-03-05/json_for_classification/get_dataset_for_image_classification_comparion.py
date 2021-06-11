# get dataset from the intersection of original annotation and croped images for image classification comparion
import argparse
from get_train_val_test_dataset_from_the_classification_files import split, label_mapping, get_train_val_test_dataset,get_image_label_list_from_origignal_annotation

from get_classification_file_with_original_annotation import  get_image_label_dict_from_original_annotation, get_classification_file_based_on_label

def main(choose_labels,original_annotation_path,croped_image_path,image_data_root,save_image_path,save_txt_folder):
    # original annotation
    original_image_label_dict= get_image_label_dict_from_original_annotation(original_annotation_path)  
    
    croped_image_label_dict= get_image_label_dict_from_original_annotation(croped_image_path)  

    label_mapping_dict=label_mapping(choose_labels)
    image_label_intersection_dict={}
    for image_name, label in croped_image_label_dict.items():
        if image_name in original_image_label_dict.keys() and int(original_image_label_dict[image_name]) in label_mapping_dict.keys() and label_mapping_dict[int(original_image_label_dict[image_name])]==int(label):
            image_label_intersection_dict[image_name]=label

    get_classification_file_based_on_label(image_label_intersection_dict,image_data_root,save_image_path)       
    image_file_path=save_image_path

    image_name_label_list=get_image_label_list_from_origignal_annotation(image_file_path,choose_labels,label_mapping_dict)
    get_train_val_test_dataset(image_name_label_list,0.7,0.333,save_txt_folder)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="get dataset intersection from original annotation and padim segment")
    parser.add_argument('--choose_labels', nargs='+', help='<Required> Set flag', default="0 5 7")
    parser.add_argument("--image_file_path",default="/git/PaDiM-master/kangqiang_result/croped_images_part_with_classification",type=str)
    parser.add_argument("--original_annotation_path",default="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification",type=str)
    parser.add_argument("--croped_image_path",default="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning",type=str)
    parser.add_argument("--image_data_root",default="/git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation/image/**/*.jpg",type=str)
    parser.add_argument("--save_image_path",default="/git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation_and_paddim_intersection_for_classification",type=str)
    parser.add_argument("--save_txt_folder",default="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_preparison_for_annotation_and_auto_croped_images_comparison",type=str)

    args=parser.parse_args()
    main(args.choose_labels,args.original_annotation_path,args.croped_image_path,args.image_data_root,args.save_image_path,args.save_txt_folder)





















