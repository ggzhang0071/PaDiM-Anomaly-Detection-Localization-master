import os,cv2, glob
#image path
image_data_root="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2/**/*.jpg"
save_image_path="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2/all_corped_images"

# json file path
json_path="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2"
image_label_name="Padim_results_image_label_for_classification.txt"

image_label_dict={}
with open(os.path.join(json_path,image_label_name),"r") as fid:
    image_label_list=fid.readlines()
    for image_label in image_label_list:
        label= image_label.split(" ")[-1].strip()
        image_name= image_label.split(" ")[0]
        image_label_dict[image_name]=label
        
image_name_list=[]
for image_name in glob.glob(image_data_root,recursive=True):
    img = cv2.imread(image_name)
    if img is None:
        print("file name isn't exists {}".format(image_name))
        os.remove(image_name)
    else:
        split_image_name_list=os.path.split(image_name)
        saved_image_name=split_image_name_list[-1]
        if image_name in image_label_dict:
            save_subfolder_path=os.path.join(save_image_path,image_label_dict[image_name])
            if not os.path.exists(save_subfolder_path):
                os.makedirs(save_subfolder_path)
            save_new_image_name=os.path.join(save_subfolder_path,saved_image_name)
        else:
            save_new_image_name=os.path.join(save_image_path,saved_image_name)

        if not os.path.exists(save_new_image_name):
            cv2.imwrite(save_new_image_name,img)


    
