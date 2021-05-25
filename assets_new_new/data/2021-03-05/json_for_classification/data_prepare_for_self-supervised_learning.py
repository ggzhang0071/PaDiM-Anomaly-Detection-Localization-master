# split data in train, val and test dataset.


import os,cv2, glob
#image path
image_data_root="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2/image/**/*.jpg"
save_image_path="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2/all_croped_images"

# json file path
json_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification"

image_label_dict={}
image_label_file_name_list=["train.txt","val.txt","test.txt"] 
for image_label_file_name in image_label_file_name_list:
    with open(os.path.join(json_path,image_label_file_name),"r") as fid:
        image_label_list=fid.readlines()
        for image_label in image_label_list:
            label= image_label.split(" ")[-1].strip()
            image_name_path= image_label.split(" ")[0]
            image_path,image_name=os.path.split(image_name_path)
            part_image_path=image_path.split("/")[5:]
            image_name_without_ext,ext=os.path.splitext(image_name)
            image_name_part=image_name_without_ext.split("_")
            recover_image_name_path=os.path.join("/".join(part_image_path),"_".join(image_name_part[:-1])+ext)
            image_label_dict[recover_image_name_path]=label
        print(len(image_label_dict.keys()))
        
image_name_list=[]
for image_name_path in glob.glob(image_data_root,recursive=True):
    img = cv2.imread(image_name_path)
    if img is None:
        print("file name isn't exists {}".format(image_name_path))
        os.remove(image_name_path)
    else:
        image_path,image_name=os.path.split(image_name_path)
        part_image_path=image_path.split("/")[5:]
        image_name_without_ext,ext=os.path.splitext(image_name)
        image_name_part=image_name_without_ext.split("_")
        recover_image_name_path=os.path.join("/".join(part_image_path),"_".join(image_name_part[:-1])+ext)
        if recover_image_name_path in image_label_dict:
            save_subfolder_path=os.path.join(save_image_path,image_label_dict[recover_image_name_path])
            if not os.path.exists(save_subfolder_path):
                os.makedirs(save_subfolder_path)
            save_new_image_name=os.path.join(save_subfolder_path,image_name)
        else:
            save_new_image_name=os.path.join(save_image_path,image_name)

        if not os.path.exists(save_new_image_name):
            cv2.imwrite(save_new_image_name,img)


    
