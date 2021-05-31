from operator import index
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
      template_img=[]
      y=[]
      anomaly_loc_and_label=[]
      image_name=[]
      original_img_shape=[]
      for item in batch:
            img.append(item[0])
            template_img.append(item[1])
            anomaly_loc_and_label.append(item[2])
            image_name.append(item[3])
            original_img_shape.append(item[4])
      img=torch.from_numpy(np.array(img)) 
      template_img=torch.from_numpy(np.array(template_img).astype("float")) 

      return img,template_img,anomaly_loc_and_label,image_name,original_img_shape


class JsonDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,json_file,image_data_root,transform=None): 
      'Initialization'
      self.anomaly_image_path_label_list=[]
      self.transform=transform
      self.anomaly_loc_and_label_list=[]
      self.image_name_list=[]
      self.template_image_name_list=[]
      self.image_data_root=image_data_root
      with open(json_file,'r') as fid:
            dataset_info = json.load(fid)
            index=-1
            for data_info in dataset_info["record"]:
                  image_path=data_info["info"]["image_path"]
                  template_image_path=data_info["info"]["template_path"]
                  if len(data_info["instances"])>0:
                        self.flag=True
                        index+=1
                        self.image_name_list.append(image_path)
                        self.template_image_name_list.append(template_image_path)
                        anomaly_loc_and_label={}
                        for i in range(len(data_info["instances"])):
                              anomaly_loc=data_info["instances"][i]["points"]
                              label=convert_NG_label(data_info["instances"][i]["distribution"],dataset_info["class_dict"])
                              anomaly_loc_and_label[anomaly_loc]=label
                        self.anomaly_loc_and_label_list.append(anomaly_loc_and_label)
                  else:
                        self.flag=False
  def __len__(self):
      'Denotes the total number of samples'
      return len(self.image_name_list)
  def __getitem__(self, idx):
      'Generates one sample of data'
      # Select sample
      img=cv2.imread(os.path.join(self.image_data_root,self.image_name_list[idx]),cv2.IMREAD_COLOR)
      template_img=cv2.imread(os.path.join(self.image_data_root,self.template_image_name_list[idx]),cv2.IMREAD_COLOR)

      if self.transform:
            original_img_shape=img.shape
            img=self.transform(image=img)
            img = img["image"].transpose(2,0,1).astype('float32')
            template_img=self.transform(image=template_img)
            template_img = template_img["image"].transpose(2,0,1).astype('float32')
      if self.flag==True:
            return img, template_img, self.anomaly_loc_and_label_list[idx], self.image_name_list[idx],original_img_shape
      else:
            return img

if __name__=="__main__":
      data_list=[]
      file_path='assets_new_new/data/2021-03-05/DFN-5X6-8L'
      Image_Data_Root = '/git/dataSet/raw/unsupervised-learning/kangqiang'
      test_file_name='test.json'
      train_sampler=10
      test_transform = Compose([
        transforms.Resize(224,224),])
      test_dataset = JsonDataset(
            json_file=os.path.join(file_path,test_file_name), 
            image_data_root=Image_Data_Root,
                        transform=test_transform,)
      test_dataloader =DataLoader(test_dataset,batch_size=128,collate_fn=collate_fn)
      for i, (img,template_img,anomaly_loc_and_label,image_name,original_img_shape) in enumerate(test_dataloader):
            if i==0:
                  print(anomaly_loc_and_label)
                  break

             
