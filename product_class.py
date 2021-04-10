import json
import os
file_name='assets/data/2021-03-05/train-aug.json'
file_path=file_name.split("/")
file_path[0]+="_new"
file_name_new=file_path.pop()

obj = json.load( open(file_name,"r"))

with open(file_name,'r') as fid:
    dataset_info = json.load(fid)

product_id_0305=[]
for data_info in dataset_info["record"]:
    product_id=data_info["info"]["product_id"]
    if product_id not in product_id_0305:
        product_id_0305.append(product_id)
print(product_id_0305)        
product_classification={}
# one product class in one filefolder
for data_info in dataset_info["record"]:
    product_id=data_info["info"]["product_id"]
    for product_class in  product_id_0305:
        if product_id==product_class:
            if product_id not in  product_classification:
                product_classification[product_id]=[]
            product_classification[product_id].append(data_info)
            
for product_id in product_classification:
    new_file_path="/".join(file_path)
    product_file_path=new_file_path+"/"+product_id
    if not os.path.exists(product_file_path):
        os.makedirs(product_file_path)
    with open(product_file_path+"/"+file_name_new, 'w+',encoding="utf8") as fid:
        json.dump(dict(product_classification[product_id]), fid)







