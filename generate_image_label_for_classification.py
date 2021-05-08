import torch,json
import os,cv2
import numpy as np
from coding_related import decode_labelme_shape
from common_lib.tools.cnn_json_related import ClassNameManager
from common_lib.tools.coding_related import decode_distribution

def convert_NG_label(distribution,class_dict):
      distribution_list=decode_distribution(distribution)
      class_name_manager=ClassNameManager(class_dict)
      NG_type=class_name_manager.get_key_from_distribution(distribution_list)
      for class_num in class_dict:
            if NG_type[-1]==class_num["class_name"]:
                  return class_num["class_id"]-4

json_file="/git/PaDiM-master/assets_new_new/data/2021-03-05"
json_file_list=["train.json","val.json","test.json"]
anomaly_image_path_label_list=[]
save_folder="json_for_classification"
save_file_list=["train.txt","val.txt","test.txt"]
for i in range(len(save_file_list)):
    with open(os.path.join(json_file,json_file_list[i]) ,'r') as fid:
        dataset_info = json.load(fid)
        for k,data_info in enumerate(dataset_info["record"]):
            anomaly_image_path_label_list.append([data_info["info"]["image_path"],convert_NG_label(data_info["instances"][0]["distribution"],dataset_info["class_dict"])])
    with open(os.path.join(json_file,save_folder,save_file_list[i]),"w+") as fid:
        fid.writelines([i[0]+" "+str(i[1])+"\n" for i in anomaly_image_path_label_list])
        fid.close()




