import random
from random import sample
import argparse
import numpy as np
import os
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
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from  coding_related import decode_labelme_shape
from config import default_config
import sys
import  copy
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader, RandomSampler

from torchvision.models import wide_resnet50_2, resnet18
#import datasets.mvtec as mvtec

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0,1,2,3' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    #parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--save_path', type=str, default='./kangqiang_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--train_num_samples', type=int, default=0)
    parser.add_argument('--test_num_samples', type=int, default=0)
    parser.add_argument('--product_class', type=str, default='0808QFN-16L')
    parser.add_argument('--threshold_coefficient', type=float, default='0.8')
    return parser.parse_args()

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
            num_workers=default_config.Num_Workers,
            pin_memory=True,
            #sampler=False,
            #drop_last=True
            )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=default_config.Batch_Size_Test,
            #shuffle=True,
            num_workers=default_config.Num_Workers,
            pin_memory=True,
            #sampler=test_sampler,
            collate_fn=collate_fn,
            #drop_last=False
            )

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for x in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to("cuda:1"))
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
        test_imgs = []
        anomaly_point_lists=[]

        # extract test set features
        for x, _, anomaly_points, loc_list in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x)
            anomaly_point_lists.extend(anomaly_points)
            #gt_list.extend(y.cpu().detach().numpy())
            #gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
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
        plot_fig(test_imgs, scores, anomaly_point_lists,save_picture_dir,save_image_dir,class_name,args.threshold_coefficient,loc_list)

    """print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")"""

    #fig.tight_layout()
    #fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

def segment_image(img,mask,save_image_dir,class_name,image_name,step=5):
    mask=mask.astype("uint8")
    mask_contour,hierarchy  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(mask_contour):
        contour_mask = cv2.fillPoly(np.zeros_like(img), [cnt], (255,255,255))
        foreground = cv2.bitwise_and(img, contour_mask)
        bounding_box=cv2.boundingRect(cnt)
        x, y, w, h = bounding_box
        #crop_region_pure_foreground = foreground[y:y+h,x:x+w]  # only fore groud
        stepx=int(w/4)
        stepy=int(h/4)
        # expand the crop range
        """if x<stepx or y<stepy or x>224-stepx or y>224-stepy:
            if x<=stepx:
                stepx=x
            if y<stepy:
                stepy=y
            if x>(224-w-stepx):
                x=224-w-stepx
            if y>(224-h-stepy):
                y=224-h-stepy"""
                
        crop_region=img[y:y+h,x:x+w] 
        crop_region_pure_foreground = foreground[y:y+h,x:x+w] 
        """for k in range(h):
            for j in range(w):
                if np.count_nonzero(crop_region_pure_foreground[k,j,:])==0:
                    crop_region_pure_foreground[k,j,:]=crop_region[k,j,:]"""
        fig=plt.figure()
        plt.imshow(crop_region_pure_foreground)
        plt.axis("off")
        save_file_name=os.path.join(save_image_dir,image_name+"_"+str(i)+".jpg")
        save_file_dir=os.path.split(save_file_name)[0]
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)
        plt.savefig(save_file_name)
        plt.close(fig)

def plot_fig(test_img, scores,anomaly_point_lists, save_picture_dir,save_image_dir,class_name,threshold_coefficient,image_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img=img/255
        img=img.permute(1, 2, 0)
        #img = denormalization(img)
        #gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        threshold=(mask.max()+mask.min())/2*args.threshold_coefficient
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        img=(255*img.numpy()).astype(np.uint8)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  
        img_for_segment=copy.deepcopy(img)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        points=anomaly_point_lists[i]
        shapes=[]
        ax_img[1].imshow(img,cmap="gray") 
        for point in points:
            shape=np.array(decode_labelme_shape(point))
            x1,y1,x2,y2 = shape[:,0].min(), shape[:,1].min(), shape[:,0].max(), shape[:,1].max()
            ax_img[1].plot([x1,x1,x2,x2,x1], [y1, y2, y2, y1, y1])
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
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
        save_file_dir=os.path.join(save_picture_dir,class_name+ '_{}'.format(i)+".jpg")
        fig_img.savefig(save_file_dir, dpi=100)
        plt.close()
        segment_image(img_for_segment,mask,save_image_dir,class_name,image_name[i])


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
    #z.cuda("cuda1,2,3")
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


if __name__ == '__main__':
    main()
