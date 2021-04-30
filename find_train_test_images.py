# split data for padim 
import os,json
def find_val_test_iamges(json_file,product_id_lists,product_classification):
    with open(json_file,"r") as fid:
        dataset_info=json.load(fid)
        class_dict_info=dataset_info['class_dict']
        for data_info in dataset_info["record"]:
                product_id=data_info["info"]["product_id"]
                if  len(data_info["instances"])>0:
                    if product_id not in product_id_lists:
                        product_id_lists.append(product_id)
                    if product_id not in product_classification["test_json_info"]:
                        product_classification["test_json_info"][product_id]=[]
                    product_classification["test_json_info"][product_id].append(data_info)
                elif  product_id in product_id_lists:
                    if  product_id not in product_classification["train_json_info"]:
                        product_classification["train_json_info"][product_id]=[]
                    product_classification["train_json_info"][product_id].append(data_info)
        return product_id_lists,product_classification, class_dict_info

if  __name__== "__main__":
    val_json='assets/data/2021-03-05/val.json'
    test_json='assets/data/2021-03-05/test.json'
    file_path='assets_new_new/data/2021-03-05'
    product_classification={"train_json_info":{},"test_json_info":{}}
    product_id_lists=[]
    file_lists=[val_json,test_json]
    for file_name in file_lists:
        product_id_lists,product_classification,class_dict_info =find_val_test_iamges(file_name,product_id_lists,product_classification)
    print(product_id_lists)
    for product_id in product_id_lists:
        file_path_new=file_path+"/"+product_id
        if not os.path.exists(file_path_new):
            os.makedirs(file_path_new)
        with open(file_path_new+"/"+"test.json", 'w+',encoding="utf8") as fid:
            json.dump({"record":product_classification["test_json_info"][product_id],"class_dict":class_dict_info},fid,indent=4)
            fid.close()
        with open(file_path_new+"/"+"train.json", 'w+',encoding="utf8") as fid:
            json.dump({"record":product_classification["train_json_info"][product_id],"class_dict":class_dict_info},fid,indent=4) 
            fid.close()

        
    
