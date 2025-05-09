from operator import index
import torch,json
import os,cv2
import numpy as np
from common_lib.tools.coding_related import decode_labelme_shape
from common_lib.tools.cnn_json_related import ClassNameManager
from common_lib.tools.coding_related import decode_distribution
from torch.utils.data import DataLoader, RandomSampler
from albumentations.augmentations import transforms
import albumentations as A

from torch.utils.data import Dataset
from torchvision import transforms


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



def collate_fn1(batch):
      img=[]
      img_rel_path=[]
      for item in batch:
            img.append(item[0])
            img_rel_path.append(item[1])

      return img,img_rel_path

class SheZhenDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(root_dir,json_path), 'r', encoding='utf-8') as f:
            self.img_list = json.load(f)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_rel_path = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        print(img_rel_path)

        return img, img_rel_path


class JsonDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,json_file,image_data_root,transform=None): 
      'Initialization'
      self.transform=transform
      self.image_data_root=image_data_root
      self.image_name_list=[]
      self.template_image_name_list=[]
      self.anomaly_loc_and_label_list=[]
      with open(json_file,'r') as fid:
            dataset_info = json.load(fid)
            index=-1
            for image_path in dataset_info:
                  template_image_path=data_info["info"]["template_path"]
                  if len(data_info["instances"])>0:
                        # if the num instance greater than 0, it is a abnormal image
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
      """data_list=[]
      file_path='/git/datasets/she_zhen_data'
      Image_Data_Root = '/git/datasets/she_zhen_data'
      test_file_name='test.json'
      train_sampler=10
      test_transform = A.Compose([
            A.SmallestMaxSize(max_size=224),
            A.CenterCrop(height=224, width=224)
            ])
      test_dataset = JsonDataset(
            json_file=os.path.join(file_path,test_file_name), 
            image_data_root=Image_Data_Root,
            transform=test_transform,)
      test_dataloader =DataLoader(test_dataset,batch_size=128,collate_fn=collate_fn)
      for i, (img,template_img,anomaly_loc_and_label,image_name,original_img_shape) in enumerate(test_dataloader):
            if i==0:
                  print(anomaly_loc_and_label)
                  break
      """
      transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
                  ])

      train_dataset = SheZhenDataset(
        json_path='train.json',
        root_dir='/git/datasets/she_zhen_data',
        transform=transform
      )

      test_dataset = SheZhenDataset(
        json_path='test.json',
        root_dir='/git/datasets/she_zhen_data',
        transform=transform
       )
      print(f"Train samples: {len(train_dataset)}")
      print(f"Test samples: {len(test_dataset)}")


      test_dataloader =DataLoader(test_dataset,batch_size=128,collate_fn=collate_fn1)
      
      for i, (img,img_rel_path) in enumerate(test_dataloader):
            if i==0:
                  print(img)
                  break
      


             
