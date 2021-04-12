import json
import os

def product_class(file_name):
    file_path=file_name.split("/")
    file_path[0]+="_new"
    os.close
    file_name_new=file_path.pop()
    obj = json.load(open(file_name,"r"))
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
    class_dict_info=dataset_info['class_dict']
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
            json.dump({"record":product_classification[product_id],"class_dict":class_dict_info}, fid,indent=4)
    return product_id_0305,class_dict_info

if __name__=="__main__":
    """train_json='assets/data/2021-03-05/train-aug.json'
    val_json='assets/data/2021-03-05/val.json'
    test_json='assets/data/2021-03-05/test.json'
    product_id_0305,class_dict_info=product_class(train_json)
    files=[train_json,val_json,test_json]
    for file_name in files:
        product_class(file_name)"""

        
    import sys
    sys.path.append("./data_loader")
    from lib.common_lib import DataManager
    from lib.data_loader import DataLoader
    train_json='assets/data/2021-03-05/train-aug.json'
    product_id_0305=['qipan-QFN-32L', 'DFN-20X20-3L', 'DFN7X6.5-6L', '0711DFN-2X3-8L', 'QFN-44L', 'QFN-28L', 'QFN-2.5X4.5-20L', 'QFN-4X4-24L', '0919DFN-10LAR', 'QFN-5X5-40L', 'QFN-3X3-H-16L', '0420QFN-4X4-24L', 'QFN3X3-20B', '1129QFN-4X4-24L', '1122QFN-3X3-16L', '1031QFN-24L', '1101QFN-40L']

    product_class=product_id_0305

    with open(train_json,'r') as fid:
        dataset_info = json.load(fid)
    class_dict_info=dataset_info["class_dict"]
    product_file_path='assets_new/data/2021-03-05'
    file_name='train-aug.json'
    for product_id in product_class:
        data = DataManager.from_json(product_file_path+"/"+product_id+"/"+file_name)
        #data = data.unique(lambda rec: rec['info']['product_id'])
        data_train,data_val=data.split(0.7)
        data_val,data_test=data_val.split(0.7)
        data_train.save_json(product_file_path+"/"+product_id+"/train.json")   
        data_val.save_json(product_file_path+"/"+product_id+"/val.json")   
        data_val.save_json(product_file_path+"/"+product_id+"/test.json")   







