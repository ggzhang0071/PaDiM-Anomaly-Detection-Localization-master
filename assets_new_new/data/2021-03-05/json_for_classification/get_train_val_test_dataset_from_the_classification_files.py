#convert the classification file to train, val and test dataset
import os,random
from abc import abstractproperty
from posixpath import splitext 

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

def label_mapping(choose_labels):
    label_mapping={}
    start_index=0
    for label in choose_labels:
        label_mapping[label]=start_index
        start_index+=1
    return  label_mapping


def get_image_label_list_from_origignal_annotation(image_file_path,choose_labels,label_mapping):
    image_name_label_list=[]
    for label in choose_labels:
        image_name_list=os.listdir(os.path.join(image_file_path,str(label)))
        print("label {} num is {}".format(label_mapping[label],len(image_name_list)))
        for image_name in image_name_list:
            image_name_label_list.append(os.path.join(str(label),image_name)+" "+str(label_mapping[label])+"\n")
    return image_name_label_list

def get_train_val_test_dataset(image_name_label_list,train_ratio,val_ratio,save_txt_folder):
    data_train_list,data_list=split(image_name_label_list,ratio=train_ratio)
    data_val_list, data_test_list=split(data_list,ratio=val_ratio)
    print("the train data num is:{}, val data num is:{}, test data num is:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))
    save_data_list=[data_train_list,data_val_list,data_test_list]
    save_data_name=["train.txt","val.txt","test.txt"]
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    for i in range(len(save_data_list)):
        with open(os.path.join(save_txt_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
            fid.writelines(save_data_list[i])
            print("{} is written".format(save_data_name[i]))

if __name__=="__main__":
    # this example is used for  self supervised learning
    image_file_path="/git/PaDiM-master/kangqiang_result/croped_images_part_with_classification"
    save_txt_folder="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning"
    choose_labels=[0,5]
    label_mapping=label_mapping(choose_labels)
    image_name_label_list=get_image_label_list_from_origignal_annotation(image_file_path,save_txt_folder,label_mapping)
    get_train_val_test_dataset(image_name_label_list,0.7,0.333)







