# get classification file with original annotation croped image to 
import os,cv2, glob
#image path where the croped images is 


# get image label dict from original annotation 

def get_image_label_dict_from_original_annotation(json_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification",image_label_file_name_list=["train.txt","val.txt","test.txt"]):  
    image_label_dict={}
    for image_label_file_name in image_label_file_name_list:
        with open(os.path.join(json_path,image_label_file_name),"r") as fid:
            image_label_list=fid.readlines()
            for image_label in image_label_list:
                label= image_label.split(" ")[-1].strip()
                image_name_path= image_label.split(" ")[0]
                image_name=os.path.split(image_name_path)[1]
                image_name_without_ext,ext=os.path.splitext(image_name)
                image_name_part=image_name_without_ext.split("_")
                recover_image_name_path="_".join(image_name_part[:-1])+ext
                image_label_dict[recover_image_name_path]=label
    return image_label_dict

# convert decentralized images to one file 
def get_classification_file_based_on_label(image_label_dict,image_data_root,save_image_path):      
    image_name_list=[]
    file_list=glob.glob(image_data_root,recursive=True)
    for image_name_path in file_list:
        img = cv2.imread(image_name_path)
        if img is None:
            print("The file name isn't exists {}".format(image_name_path))
            os.remove(image_name_path)
        else:
            image_name=os.path.split(image_name_path)[-1]
            image_name_without_ext,ext=os.path.splitext(image_name)
            image_name_part=image_name_without_ext.split("_")
            recover_image_name_path="_".join(image_name_part[:-1])+ext
            if recover_image_name_path in image_label_dict:
                save_subfolder_path=os.path.join(save_image_path,image_label_dict[recover_image_name_path])
                if not os.path.exists(save_subfolder_path):
                    os.makedirs(save_subfolder_path)
                save_new_image_name=os.path.join(save_subfolder_path,image_name)
            else:
                save_new_image_name=os.path.join(save_image_path,image_name)
            if not os.path.exists(save_new_image_name):
                cv2.imwrite(save_new_image_name,img)
            else:
                print("The image {} is existed".format(image_name))

if __name__=="__main__":
    image_data_root="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2/image/**/*.jpg"

    # image path where the classification results is saved
    save_image_path="/git/PaDiM-master/kangqiang_result/croped_images_with_original_annotation_for_classification"

    # json file path with original anotation
    json_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification"  

    image_label_dict= get_image_label_dict_from_original_annotation(json_path)  
    get_classification_file_based_on_label(image_label_dict,image_data_root,save_image_path)
   



    
