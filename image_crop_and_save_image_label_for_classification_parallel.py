# crop image based on the annotation information and save the image and label information
from dataset  import JsonDataset
from common_lib.tools.coding_related import decode_distribution
#from common_lib import decode_labelme_shape
from coding_related import decode_labelme_shape
from common_lib.tools.cnn_json_related import ClassNameManager
import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def convert_NG_label(distribution,class_dict):
      distribution_list=decode_distribution(distribution)
      class_name_manager=ClassNameManager(class_dict)
      NG_type=class_name_manager.get_key_from_distribution(distribution_list)
      for class_num in class_dict:
            if NG_type[-1]==class_num["class_name"]:
                  return class_num["class_id"]-4
#image_data_root = '/workspace/dataSet/raw/unsupervised-learning/kangqiang/'

def image_crop(data):
    anomaly_image_path_label_list=[]
    for i, rec in enumerate(data):
        image_name=rec['info']['image_path']
        image_dir,image_name_part=os.path.split(image_name)
        img = cv2.imread(os.path.join(image_data_root,image_name))[...,::-1]
        assert img is not None, rec
        for p,inst in enumerate(rec['instances']):
            save_dir=os.path.join(save_image_root, image_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            save_new_image_name=os.path.join(save_dir, os.path.splitext(image_name_part)[0]+"_"+str(p)+".jpg")
            if not os.path.exists(save_new_image_name):
                shapes = np.array(decode_labelme_shape(inst['points']))
                x1,y1,x2,y2 = int(shapes[:,0].min()), int(shapes[:,1].min()), int(shapes[:,0].max()), int(shapes[:,1].max())
                crop_img=img[y1:y2,x1:x2]
                plt.imshow(crop_img) 
                plt.axis("off") 
                # save new images
                plt.imsave(save_new_image_name,crop_img) 
            # delete the corrupted file
            img1 = cv2.imread(save_new_image_name)
            if img1 is None:
                print(save_new_image_name)
                os.remove(save_new_image_name)
                continue
            if [save_new_image_name,convert_NG_label(rec['instances'][0]['distribution'],dataset_info["class_dict"])] in anomaly_image_path_label_list:
                pass
            else:
                anomaly_image_path_label_list.append([save_new_image_name,convert_NG_label(rec['instances'][0]['distribution'],dataset_info["class_dict"])])
    return anomaly_image_path_label_list

def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

if __name__ == '__main__':
    from multiprocessing import Pool
    image_data_root="/git/dataSet/raw/unsupervised-learning/kangqiang/"
    save_image_root="/git/PaDiM-master/kangqiang_result/croped_images"
    json_path='assets_new_new/data/2021-03-05/json_for_classification'
    #save_folder="json_for_classification"
    json_file_list=["train.json","val.json","test.json"]
    save_file_list=["train.txt","val.txt","test.txt"]
    data_list=[]
    for json_file in json_file_list:
        with open(os.path.join(json_path,json_file)) as fid:
            dataset_info=json.load(fid)
            data_list.append(dataset_info['record'])

    cpu_num=24
    for k, data in enumerate(data_list):
        print("the current process is {} dataset".format(json_file_list[k]))
        with Pool(cpu_num) as p:
            anomaly_image_path_label_list=p.map(image_crop, list(func(data, cpu_num)))
        print(len(anomaly_image_path_label_list[0]))
        print(os.path.join(json_path,save_file_list[k]))
        with open(os.path.join(json_path,save_file_list[k]),"w+") as fid:
            num=0
            for i in  range(len(anomaly_image_path_label_list)):
                for j in range(len(anomaly_image_path_label_list[i])):
                    num+=1
                    fid.writelines(anomaly_image_path_label_list[i][j][0]+" "+str(anomaly_image_path_label_list[i][j][1])+"\n")
    


      