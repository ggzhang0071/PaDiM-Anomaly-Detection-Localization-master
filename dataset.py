import torch,json
import os,cv2
import numpy as np

class JsonDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,json_file,image_data_root,transform=None): 
      'Initialization'
      self.loc_lists=[]
      self.transform=transform
      self.anomaly_loc_lists=[]
      self.image_data_root=image_data_root
      with open(json_file,'r') as fid:
            dataset_info = json.load(fid)
            for k,data_info in enumerate(dataset_info["record"]):
                  image_path=data_info["info"]["image_path"]
                  self.loc_lists.append(image_path)
                  if len(data_info["instances"])>0:
                        self.flag=True
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
            anomaly_points=self.anomaly_loc_lists[idx]
            return [img,anomaly_points]
        else:
            return  img