import random
from random import sample
import argparse
from typing_extensions import runtime
from PIL.Image import CONTAINER
from albumentations.augmentations.functional import crop_keypoint_by_coords
from matplotlib import image
import numpy as np
import os
import plantcv.plantcv as pcv
from numpy.lib.function_base import diff
from scipy.ndimage.measurements import label
import imutils
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from dataset import JsonDataset, collate_fn
from skimage import morphology
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from skimage.segmentation import mark_boundaries, watershed
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
import matplotlib.pyplot as plt
import matplotlib
from  coding_related import decode_labelme_shape
from config import default_config
import sys
import  copy
import torch
import torch.nn.functional as F
import cv2
import onnx
from torch.utils.data import DataLoader, RandomSampler
from scipy import ndimage as ndi
from torchvision.models import wide_resnet50_2, resnet18
#import datasets.mvtec as mvtec

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0,1,3' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    #parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--save_path', type=str, default='/git/PaDiM-master/kangqiang_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--train_num_samples', type=int, default=10)
    parser.add_argument('--test_num_samples', type=int, default=30)
    parser.add_argument('--product_class', type=str, default='0808QFN-16L')
    parser.add_argument('--threshold_coefficient', type=float, default=0.8)
    parser.add_argument('--gaussian_blur_kernel_size', type=int, default=5)
    return parser.parse_args()
def tensorrt_optimize_model(input,original_model):
    input_name = ['input']
    output_name = ['output']
    input=input.to("cuda")
    original_model.to("cuda")
    torch.onnx.export(original_model, input, 'temp_wide_resnet50_2.onnx', input_names=input_name, output_names=output_name, verbose=True)
    model = onnx.load('temp_wide_resnet50_2.onnx')
    onnx.checker.check_model(model)
    print("==> Passed")
    return model

def crop_image(img,bounding_box):
    x, y, w, h = bounding_box
    height,width=img.shape[:2]
    stepx=int(w/4)
    stepy=int(h/4)
    # expand the crop range
    if x<=stepx:
        startx=0
    else:
        startx=x-stepx
    if y<=stepy:
        starty=0
    else:
        starty=y-stepy
    if x+w+stepx>=width:
        endx=width
    else:
        endx=x+w+stepx
    if y+h+stepy>=height:
        endy=height
    else:
        endy=y+h+stepy
    crop_images=img[starty:endy,startx:endx]
    return crop_images


def main():
    args = parse_args()
    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    """
    os.removedirs(args.save_path+"/pictures_wide_resnet50_2")
    os.removedirs(args.save_path+"/pictures_temp_wide_resnet50_2")
    os.remove(args.save_path+"/pictures_temp_wide_resnet50_2/*")"""

    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    
    # load dataset
    train_transform = Compose([
        transforms.Resize(224, 224),
    ])

    val_transform = Compose([
        transforms.Resize(224,224),
    ])
    
    file_path='assets_new_new/data/2021-03-05'
    train_file_name='train.json'
    val_file_name="test.json"
    product_class=[args.product_class]
    for class_name in product_class:
        train_dataset = JsonDataset(
            json_file=file_path+"/"+class_name+"/"+train_file_name,
            image_data_root=default_config.Image_Data_Root,
            transform=train_transform,
        )
        test_dataset = JsonDataset(
            json_file=file_path+"/"+class_name+"/"+val_file_name,
            image_data_root=default_config.Image_Data_Root,
            transform=val_transform,
            )  # use real data as validation
        if args.train_num_samples==0:
            train_sampler=None
        else:
           train_sampler=RandomSampler(train_dataset,num_samples=args.train_num_samples, replacement=True)

        if args.test_num_samples==0:
            test_sampler=None
        else:
            test_sampler=RandomSampler(test_dataset, num_samples=args.test_num_samples, replacement=True)

        train_dataloader =DataLoader(train_dataset,
        batch_size=default_config.Batch_Size,
            #shuffle=True,
            #pin_memory=True,
            sampler=train_sampler,
            drop_last=True
            )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=default_config.Batch_Size_Test,
            #shuffle=False,
            #pin_memory=True,
            sampler=test_sampler,
            collate_fn=collate_fn,
            drop_last=False
            )

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for  x in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
   
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)
    

        gt_list = []
        gt_mask_list = []
        test_img_list= []
        template_img_list=[]
        anomaly_point_lists=[]
        anomaly_loc_and_label_list=[]
        image_name_list=[]
        original_img_shape_list=[]
        # extract test set features
        for kk, (x, template_img, anomaly_loc_and_label, image_name,original_img_shape) in tqdm(enumerate(test_dataloader), '| feature extraction | test | %s |' % class_name):
            test_img_list.extend(x)
            template_img_list.extend(template_img)
            anomaly_loc_and_label_list.extend(anomaly_loc_and_label)
            image_name_list.extend(image_name)
            original_img_shape_list.extend(original_img_shape)

            #gt_list.extend(y.cpu().detach().numpy())
            #gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            
            with torch.no_grad():
                _ = model(x.to(device))
            """if  kk==0:
                model=tensorrt_optimize_model(torch.stack(test_img_list),model)"""
           
            # get intermediate layer outputs
    
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        # for disturb anlysis, this can be deleted?
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=1)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        """gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        #gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]"""

        """# calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))"""

        #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_picture_dir = args.save_path + '/' + f'pictures_{args.arch}'
        save_image_dir=args.save_path + '/' + f'segment_image_result_{args.arch}'
        
        os.makedirs(save_picture_dir, exist_ok=True)
        os.makedirs(save_image_dir, exist_ok=True)
        plot_fig(test_img_list,template_img_list,anomaly_loc_and_label_list,original_img_shape_list,scores, anomaly_point_lists,save_picture_dir,save_image_dir,class_name,args.threshold_coefficient,image_name_list)

    """print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")"""

    #fig.tight_layout()
    #fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

def grabcut_image_segment(image,mask):
    fgModel = np.zeros((1, 65), dtype="float")  #两个空数组，便于在从背景分割前景时使用（fgModel和bgModel）
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the bounding box segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, None,None, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
    values = (
      ("Definite Background", cv2.GC_BGD),
      ("Probable Background", cv2.GC_PR_BGD),
      ("Definite Foreground", cv2.GC_FGD),
      ("Probable Foreground", cv2.GC_PR_FGD),
    )
    # loop over the possible GrabCut mask values
    for (name, value) in values:
        # construct a mask that for the current value
        #print("[INFO] showing mask for '{}'".format(name))
        valueMask = (mask == value).astype("uint8") * 255
        # display the mask so we can visualize it
        #ax.imshow(valueMask.astype("uint8"))
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)
    return output

def segment_image(img,template_img,heat_map,mask,anomaly_loc_and_label,original_img_shape,save_image_dir,class_name,image_name):
    height,width=original_img_shape[:2]
    img=cv2.resize(img,(height,width))
    template_img=template_img.numpy().astype("uint8")
    template_img=template_img.transpose(1,2,0)
    template_img=cv2.resize(template_img,(height,width))
    heat_map=cv2.resize(heat_map,(height,width))
    mask=cv2.resize(mask,(height,width))

    image_label_for_classifiation=[]
    mask=mask.astype("uint8")
    mask,mask_contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for i, cnt in enumerate(mask_contour):
        # image segment from predicted mask
        mask_predicted = cv2.fillPoly(np.zeros_like(img), [cnt], (255,255,255))
        for point,label in anomaly_loc_and_label.items():
            shape=np.array(decode_labelme_shape(point))
            shape=shape.astype("uint8")
            x1,y1,x2,y2 = shape[:,0].min(), shape[:,1].min(), shape[:,0].max(), shape[:,1].max()  
            mask_annotation,binary, contours_annotation,hierarchy=pcv.rectangle_mask(img=img,p1=(x1,y1),p2=(x2,y2),color="black")
            mask_combined=cv2.bitwise_and(mask_annotation,mask_predicted)
            mask_combined=cv2.cvtColor(mask_combined,cv2.COLOR_BGR2GRAY)
            rect,binary_mask_combined=cv2.threshold(mask_combined,5,255,cv2.THRESH_BINARY)
            binary_mask_combined,contour_combined,obj_hierarchy=cv2.findContours(binary_mask_combined,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_area_threshold=2
            if len(contour_combined)>0:
                contour_area_comblined=cv2.contourArea(contour_combined[0])*contour_area_threshold
            else:
                continue
            if contours_annotation==[]:
                continue

            if contour_area_comblined>= cv2.contourArea(contours_annotation[0]) or contour_area_comblined>=cv2.contourArea(cnt): 
                foreground = cv2.bitwise_and(img, mask_predicted)
                bounding_box=cv2.boundingRect(cnt)
                x, y, w, h = bounding_box
                #croped_region_pure_foreground = foreground[y:y+h,x:x+w]  # only fore groud
                #get maskfrom heat map      
                small_mask=get_mask_from_heat_map(heat_map)
                small_mask_pure_foreground=crop_image(small_mask,bounding_box)
                croped_region_pure_foreground=crop_image(img,bounding_box)
                croped_template_img=crop_image(template_img,bounding_box)
                if w>7 and h>7:
                    if  small_mask_pure_foreground.size != 0 and small_mask_pure_foreground.max()!=0:
                        croped_region_pure_foreground=grabcut_image_segment(croped_region_pure_foreground,small_mask_pure_foreground)
                    #print(croped_region_pure_foreground.shape,croped_template_img.shape)
                    flag, diff_mask,bounding_box_diff=get_mask_from_backgroud_subtraction(croped_region_pure_foreground,croped_template_img)
                    if flag==1:
                        croped_region_pure_foreground=crop_image(croped_region_pure_foreground,bounding_box_diff)
                        croped_diff_mask=crop_image(diff_mask,bounding_box_diff)
                        croped_region_pure_foreground = cv2.bitwise_and(croped_region_pure_foreground,croped_diff_mask)
                if croped_region_pure_foreground.size>0:
                    plt.imshow(croped_region_pure_foreground[:,:,::-1])
                    plt.axis("off")
                    save_file_name=os.path.join(save_image_dir,image_name)
                    save_file_dir,save_image_name=os.path.split(save_file_name)
                    save_file_name_without_ext=os.path.splitext(save_image_name)[0]
                    if not os.path.exists(save_file_dir):
                        os.makedirs(save_file_dir)
                    save_image_name_new=os.path.join(save_file_dir,save_file_name_without_ext+"_"+str(i)+".jpg")
                    plt.imsave(save_image_name_new,croped_region_pure_foreground)
                    image_name_label_list=[save_image_name_new,label]
                    if image_name_label_list not in image_label_for_classifiation:
                        image_label_for_classifiation.append(image_name_label_list)
                
    with open(os.path.join(save_image_dir,"Padim_segment_image_name_label_for_classification.txt"),"a+") as fid:
                        fid.writelines([save_image_name_new+" "+str(i[1])+"\n" for i in image_label_for_classifiation])
                        

def get_threshold_from_hist(heat_map):
    hist = cv2.calcHist([heat_map],[0],None,[256],[0,256])
    num_sum=0
    for i in range(256):
        num_sum+=hist[i]
        if num_sum>=224*224*0.9:
            return i
def get_mask_from_heat_map(heat_map):
    heat_map = cv2.normalize(heat_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    threshold=get_threshold_from_hist(heat_map)
    img1 = cv2.GaussianBlur(heat_map,(3,3),0)
    _,Thr_img = cv2.threshold(heat_map,threshold,255,cv2.THRESH_BINARY)#设定红色通道阈值210（阈值影响梯度运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))         #定义矩形结构元素
    gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel) #梯度
    histr = cv2.calcHist([gradient],[0],None,[256],[0,256])
    mask=np.zeros_like(gradient,dtype="uint8")
    threshold=0
    mask[gradient < threshold] = 1
    return mask

def preprocess_image(img):
    #bilaterar was use to reduce noise
    #bilateral_filtered_image = cv2.bilateralFilter(image,5, 150, 150)
    if img.shape[-1]!=3:
        img=img.transpose(1,2,0)
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_image=cv2.GaussianBlur(gray_image,(5, 5),0)
    return gray_image

def get_mask_from_backgroud_subtraction(image,template_image):
    img=preprocess_image(image)
    template_img=preprocess_image(template_image)
    image_diff=cv2.absdiff(img, template_img)
    # morphology to find contours
    kernel = np.ones((5,5),np.uint8)
    close_operated_image_diff = cv2.morphologyEx(image_diff, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresholded, 5)
    median,contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==1:
        max_countour = max(contours, key = cv2.contourArea)
        bounding_box=cv2.boundingRect(max_countour)
        area=cv2.contourArea(max_countour)
        diff_mask = cv2.fillPoly(np.zeros_like(img), [max_countour], (255,255,255))
        diff_mask_3d=np.expand_dims(diff_mask,0).repeat(3,axis=0).transpose(1,2,0)
        return 1,diff_mask_3d,bounding_box
    else:
        return 0, None,None

    """cv2.drawContours(image_diff, max_countour, -1, (0, 0, 255),1)
    fig_img, ax_img = plt.subplots(1, 2, figsize=(8, 4))
    fig_img.subplots_adjust(right=0.9)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img[::,::-1])
        ax_img[0].title.set_text('original image')
        ax_img[1].imshow(image_diff[::,::-1]) 
        ax_img[1].title.set_text('image diff')
    plt.savefig("original_image_and_image_diff.jpg")

    return  diff_mask_3d"""
    
def plot_fig(test_img,template_img_list,anomaly_loc_and_label_list,original_img_shape_list,scores,anomaly_point_lists, save_picture_dir,save_image_dir,class_name,threshold_coefficient,image_name_list):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img=img.numpy().astype("uint8")
        """contours=get_contours_from_backgroud_subtraction(img,template_img)
        img=cv2.GaussianBlur(img,(5, 5),0)
        template_img=cv2.GaussianBlur(template_img,(5, 5),0)
        image_subtraction_diff = cv2.absdiff(img, template_img)
        normalized_image_diff = cv2.normalize(image_subtraction_diff, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        _,diff_mask = cv2.threshold(normalized_image_diff,30,255,cv2.THRESH_BINARY)
        diff_mask=diff_mask.transpose(1,2,0)"""

        img=img/255
        img=img.transpose(1, 2, 0)
        #img = denormalization(img)
        #gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]

        threshold_coefficient=0.8
        threshold=(mask.max()+mask.min())/2*threshold_coefficient
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(1, 5, figsize=(15, 3))
        fig_img.subplots_adjust(right=0.9)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False) 
        img=(255*img).astype(np.uint8)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img_for_segment=copy.deepcopy(img)
        
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        height,width=original_img_shape_list[i][:2]
        original_img=cv2.resize(copy.deepcopy(img),(height,width))
        ax_img[1].imshow(original_img,cmap="gray")
        annomaly_points_label=anomaly_loc_and_label_list[i]
        for point, _ in annomaly_points_label.items():
            shape=np.array(decode_labelme_shape(point))
            x1,y1,x2,y2 = shape[:,0].min(), shape[:,1].min(), shape[:,0].max(), shape[:,1].max()
            ax_img[1].plot([x1,x1,x2,x2,x1], [y1, y2, y2, y1, y1])

        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        save_file_name=os.path.join(save_picture_dir,class_name,image_name_list[i])
        save_file_dir,save_image_name=os.path.split(save_file_name)
        save_file_name_without_ext=os.path.splitext(save_image_name)[0]
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)
        save_image_name_new=os.path.join(save_file_dir,save_file_name_without_ext+".jpg")
        plt.savefig(save_image_name_new)
        segment_image(img_for_segment,template_img_list[i],heat_map,mask,anomaly_loc_and_label_list[i],original_img_shape_list[i],save_image_dir,class_name,image_name_list[i])

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
    main()

