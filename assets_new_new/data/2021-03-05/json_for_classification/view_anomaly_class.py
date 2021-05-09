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
classes={"0": "lianjiao", "1": "shakong", "2": "guoshi", "3": "huashang", "4":"yanghua", "5":"tongheidian", "6":"heidian", "7":"feilinjiao", "8":"wuzi", "9":"yiwu"}
class_list=[]
for key,value in class_dict.items():
    class_list.append((classes[key],value))

class_list=sorted(class_list,key=lambda x:(x[1],x[0]))
print(class_list)


file_list=[]
with open("train.txt",'r') as fid:
    lines=fid.readlines()
    for line in lines:
        label= line.split(" ")[-1].strip()
        #[7, 1,4,3]
        if int(label)==7:
            file_list.append(line.split(" ")[0])
print(file_list)



    
