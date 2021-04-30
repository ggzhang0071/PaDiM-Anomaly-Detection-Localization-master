import torch,json
import os,cv2
import numpy as np
from coding_related import decode_labelme_shape
from common_lib.tools.cnn_json_related import ClassNameManager
from common_lib.tools.coding_related import decode_distribution
from torch.utils.data import DataLoader, RandomSampler
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms


def convert_NG_label(distribution,class_dict):
      distribution_list=decode_distribution(distribution)
      class_name_manager=ClassNameManager(class_dict)
      NG_type=class_name_manager.get_key_from_distribution(distribution_list)
      for class_num in class_dict:
            if NG_type[-1]==class_num["class_name"]:
                  return class_num["class_id"]-4

def collate_fn(batch):
      img=[]
      y=[]
      points=[]
      image_name=[]
      for item in batch:
            img.append(item[0])
            y.append(item[1])
            points.append(item[2])
            image_name.append(item[3])
      img=torch.from_numpy(np.array(img)) 
      return img,y,points,image_name

class JsonDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,json_file,image_data_root,transform=None): 
      'Initialization'
      self.anomaly_image_path_label_list=[]
      self.transform=transform
      self.anomaly_loc_lists=[]
      self.loc_lists=[]
      self.label=[]
      self.image_data_root=image_data_root
      with open(json_file,'r') as fid:
            dataset_info = json.load(fid)
            for k,data_info in enumerate(dataset_info["record"]):
                  image_path=data_info["info"]["image_path"]
                  if len(data_info["instances"])>0:
                        self.flag=True
                        self.loc_lists.append(image_path)
                        self.label.append(convert_NG_label(data_info["instances"][0]["distribution"],dataset_info["class_dict"]))

                        self.anomaly_loc_lists.append([])
                        for i in range(len(data_info["instances"])):
                              self.anomaly_loc_lists[k].append(data_info["instances"][i]["points"])
                  else:
                        self.flag=False
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.loc_lists)
  def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        img=cv2.imread(os.path.join(self.image_data_root,self.loc_lists[idx]),cv2.IMREAD_COLOR)
        if self.transform:
            img=self.transform(image=img)
        img = img["image"].transpose(2,0,1).astype('float32')
        if self.flag==True:
            return img, self.label[idx], self.anomaly_loc_lists[idx], self.loc_lists[idx]
        else:
            return  img

if __name__=="__main__":
      data_list=[]
      file_path='assets_new_new/data/2021-03-05/DFN-5X6-8L'
      Image_Data_Root = '/git/dataSet/raw/unsupervised-learning/kangqiang'
      train_file_name='test.json'
      train_sampler=10
      train_transform = Compose([
        transforms.Resize(224, 224),])
      test_transform = Compose([
        transforms.Resize(224,224),])
      train_dataset = JsonDataset(
            json_file=os.path.join(file_path,train_file_name), 
            image_data_root=Image_Data_Root,
                        transform=test_transform,)
      train_dataloader =DataLoader(train_dataset,batch_size=128,collate_fn=collate_fn)
      for i, (images,anomaly_points,image_name,label) in enumerate(train_dataloader):
            print(len(images))
             
