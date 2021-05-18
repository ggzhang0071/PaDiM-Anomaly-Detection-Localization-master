import os
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

json_path="/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification"
save_folder="specific_label_dataset_produce_from_original_annotation"
image_label_file_list=["train.txt","val.txt", "test.txt"]
choose_label=["8","9"]
label_mapping={"8":"0","9":"1"}
image_for_specific_label_list=[]
for file_name in image_label_file_list:
    with open(os.path.join(json_path,file_name)) as fid:
        image_label_list=fid.readlines()
        for image_label in image_label_list:
            label= image_label.split(" ")[-1].strip()
            image_name= image_label.split(" ")[0]
            if label in choose_label:
                image_for_specific_label_list.append(image_name+" "+label_mapping[label]+"\n")

data_train_list,data_list=split(image_for_specific_label_list,ratio=0.7)
data_val_list, data_test_list=split(data_list,ratio=0.333)
print("train data num:{}, val data num:{}, test data num:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))


save_data_list=[data_train_list,data_val_list,data_test_list]
save_data_name=["train.txt","val.txt","test.txt"]
for i in range(len(save_data_list)):
    with open(os.path.join(json_path,save_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
        fid.writelines(save_data_list[i])
        print("{} is written".format(save_data_name[i]))

        

