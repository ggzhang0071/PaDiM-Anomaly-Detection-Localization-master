import os,copy
from coding_related import decode_labelme_shape
from common_lib.tools.cnn_json_related import ClassNameManager
from common_lib.tools.coding_related import decode_distribution
import json

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

#get_key_from_distribution
product_class=['0708DFN-8L', 'DFN-5X6-T-8L', '0420QFN-5X6-8L' ,'1101QFN-40L', '0713DFN-2X3-8L', 'DFN-5X6-8L' ,'1129QFN-4X4-24L','QFN-3X3-16L']

save_folder="json_for_classification"
json_path="assets_new_new/data/2021-03-05"

data_list=[]
for i, product_id in enumerate(product_class):
    product_id_json=os.path.join(json_path,product_id,"test.json")
    with open(product_id_json,'r') as fid:
        dataset_info=json.load(fid)
    for info_item in dataset_info["record"]:
        data_list.append(info_item)

data_train_list,data_list=split(data_list,ratio=0.7)
data_val_list, data_test_list=split(data_list,ratio=0.333)
print("train data num:{}, val data num:{}, test data num:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))


save_data_list=[data_train_list,data_val_list,data_test_list]
save_data_name=["train.json","val.json","test.json"]
for i in range(len(save_data_list)):
    with open(os.path.join(json_path,save_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
        json.dump({"record":save_data_list[i],"class_dict":dataset_info['class_dict']},fid,indent=4)
        print("{} is written".format(save_data_name[i]))



    


