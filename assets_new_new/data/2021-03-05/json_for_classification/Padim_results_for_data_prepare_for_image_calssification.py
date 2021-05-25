import json
import os,random,cv2
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.arrayprint import printoptions

def split(full_list,shuffle=True,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

product_id=["0420QFN-5X6-8L", "0708DFN-8L",  "0713DFN-2X3-8L",  "1101QFN-40L",  "1129QFN-4X4-24L", "DFN-5X6-8L ", "DFN-5X6-T-8L", "QFN-3X3-16L"]

classes={"0": "lianjiao", "1": "shakong", "2": "guoshi", "3": "huashang", "4":"yanghua", "5":"tongheidian", "6":"heidian", "7":"feilinjiao", "8":"wuzi", "9":"yiwu"}

json_path="/git/PaDiM-master/kangqiang_result/segment_image_result_wide_resnet50_2"
image_name="Padim_results_image_label_for_classification.txt"
save_folder="dataset_produce_from_padim"

choose_label=[8,9]
label_mapping={"8":"0","9":"1"}
part_image_label_list=[]
show_size=3
num=1
with open(os.path.join(json_path,image_name),"r") as fid:
    image_label_list=fid.readlines()
    for image_label in image_label_list:
        label= image_label.split(" ")[-1].strip()
        image_name= image_label.split(" ")[0]
        image_split_path_list=image_name.split("/")
        if image_split_path_list[4] not in product_id:
            continue
        if int(label) in choose_label:
            part_image_label_list.append(image_name+" "+label_mapping[label]+"\n")


data_train_list,data_list=split(part_image_label_list,ratio=0.7)
data_val_list, data_test_list=split(data_list,ratio=0.333)
print("train data num:{}, val data num:{}, test data num:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))


save_data_list=[data_train_list,data_val_list,data_test_list]
save_data_name=["train.txt","val.txt","test.txt"]
for i in range(len(save_data_list)):
    with open(os.path.join(save_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
        fid.writelines(save_data_list[i])
        print("{} is written".format(save_data_name[i]))
  
        
            