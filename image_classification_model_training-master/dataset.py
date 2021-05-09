from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json,os
import torch

f = open('config.json')
config = json.load(f)

input_size = config['input_size']

transform_train = transforms.Compose([
    transforms.Resize((input_size,input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize((input_size,input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# rescale longer side to to input_size and zero pad the shorter side to input_size
# img is a Pillow Image object
def scale_and_pad(img):
    w, h = img.size
    scale = input_size / max(w, h)
    if w > h:
        img = img.resize((input_size, int(h*scale)))
    else:
        img = img.resize((int(w*scale), input_size))
    img = np.array(img)
    canvas = np.zeros((input_size, input_size, 3))
    start_idx = (input_size-min(img.shape[:2])) // 2
    if img.shape[0] == input_size:
        canvas[:, start_idx:start_idx+img.shape[1], :] = img
    else:
        canvas[start_idx:start_idx+img.shape[0], :, :] = img
    return Image.fromarray(np.uint8(canvas))


class CustomDataset(Dataset):
    def __init__(self, txt_file,image_data_root,transform):
        f = open(txt_file, 'r')
        lines = f.readlines()
        self.image_data_root=image_data_root
        self.transform = transform
        self.x = []
        self.y = []
        for line in lines:
            d = line.split(' ')
            img_path, label = d[0], int(d[1])
            self.x += [img_path]
            self.y += [label]
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_data_root,self.x[index])).convert("RGB")
        #img = scale_and_pad(img)
        img = self.transform(img)
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img, self.y[index]


class OOD_dataset(Dataset):
    def __init__(self, txt_file, transform):
        f = open(txt_file, 'r')
        lines = f.readlines()
        self.transform = transform
        self.x = []
        for line in lines:
            self.x += [line.split(' ')[0]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_data_root,self.x[index])).convert("RGB")
        img = self.transform(img)
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img


# generate 3 channel Gaussian noise data
class Gaussian_noise_dataset(Dataset):
    def __init__(self, input_size, num_of_data):
        self.num_of_data = num_of_data
    
    def __len__(self):
        return self.num_of_data

    def __getitem__(self, index):
        return torch.from_numpy(np.random.normal(size=(3,input_size,input_size))).float()


if __name__ == '__main__':
    file = '/git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/val.txt'
    f = open(file)
    lines = f.readlines()
    for line in lines:
        img_path = line.split(' ')[0]
        #img = Image.open(img_path)
        img = Image.open('/git/PaDiM-master/kangqiang_result/croped_images/image/1101QFN-40L/lot1/NG/ZM/1101QFN-40L-976-53_0_0_6639_18170_0.jpg')

        img = transform_test(img)
        if img.shape[0] != 3:
            print(img.shape)

