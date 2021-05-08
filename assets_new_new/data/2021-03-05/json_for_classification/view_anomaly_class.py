import os,collections
json_path='assets_new_new/data/2021-03-05/json_for_classification'
class_dict={}
with open("train.txt",'r') as fid:
    lines=fid.readlines()
    for line in lines:
        label= line.split(" ")[-1].strip()
        if label not in class_dict:
            class_dict[label]=1
        else:
            class_dict[label]+=1
class_list=sorted(class_dict.items(),key=lambda x:(x[1],x[0]))
print(class_list)




    
