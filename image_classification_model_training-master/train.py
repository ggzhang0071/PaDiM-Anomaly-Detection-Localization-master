import torch
import numpy as np
import json
from models.resnet import *
from models.downstream_net import *
from dataset import transform_train, transform_test, CustomDataset, OOD_dataset
from sklearn import metrics
import torch.utils.data as data
from tqdm import tqdm
import pandas as pd
import os
from loss.focal_loss import FocalLoss
from loss.AMSoftmax_loss import AMSoftmax
from loss.ood_gaussian_loss import *
import time
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

f = open('config_part_label.json')
#f = open('config.json')
config = json.load(f)

epochs = config['epochs']
batch_size = config['batch_size']
class_num = config['class_num']
embedding_dim = config['embedding_dim']
classes = config['classes']
# convert classes key from string to int
classes = {int(key):classes[key] for key in classes}
#device_id = config['device_id']
json_dir=config['json_dir']
train_path = os.path.join(json_dir,config['tain_file'])
val_path=os.path.join(json_dir,config['val_file']) 
test_path=os.path.join(json_dir,config['test_file'])
image_data_root=config['image_data_root']
save_path=config['save_path']
if not os.path.exists(save_path):
    os.makedirs(save_path)

#torch.cuda.set_device(device_id)

args = {}
args['lr'] = 0.001
args['momentum'] = 0.9
args['weight_decay'] = 5e-4

use_cuda = torch.cuda.is_available()


############################### model construction #################################################
model_dim = 2048
model = resnet50(pretrained=True)
model.fc = net(model_dim, embedding_dim, class_num)
###########################################################################################################


if use_cuda:
    #model = model.cuda()
    model = torch.nn.DataParallel(model)
    model=model.cuda()

model.eval()

print('training with:', model.__class__.__name__)
print('total', sum(p.numel() for p in model.parameters() if p.requires_grad), 'trainable parameters')
print()
print('=======================================================================================================')
print('BE ADVISED! training results will be written into the following path:')
print(save_path)
print('terminate the process now if the save path is incorrect!')
print('=======================================================================================================')
print()

def get_embeddings_from_file(data_path):
    dic = {}
    f = open(data_path, 'r')
    for line in f:
        file_path = line.split(' ')[0]
        label = int(line.split(' ')[1])
        if label not in dic:
            dic[label] = [file_path]
        else:
            dic[label] += [file_path]

    for k in dic:
        random.shuffle(dic[k])

    dic = collections.OrderedDict(sorted(dic.items()))

    num_list = [len(dic[k]) for k in dic]
    embeddings = {}
    for k in tqdm(dic, desc='transforming embedding'):
        for file_path in dic[k]:
            image = Image.open(file_path)
            image = transform_test(image).unsqueeze(0).cuda()
            logits, out = model(image)
            out = np.array(out.squeeze().detach().cpu())
            if k not in embeddings:
                embeddings[k] = [out]
            else:
                embeddings[k] += [out]
    for key in embeddings:
        embeddings[key] = np.array(embeddings[key])
    return embeddings

# deprecated
'''
def get_gaussian_kernels(embeddings):
    gaussians = {}
    for key in embeddings:
        gaussians[key] = (np.mean(embeddings[key], axis=0).tolist(), np.cov(embeddings[key], rowvar=0).tolist())

    f = open('embedding_gaussians_64.json', 'w')
    json.dump(gaussians, f, indent=4)

    mus = []
    covs = []
    for key in gaussians:
        mus += [gaussians[key][0]]
        covs += [gaussians[key][1]]
    mus = torch.from_numpy(np.array(mus)).cuda(device_id)
    covs = torch.from_numpy(np.array(covs)).cuda(device_id)
    kernels = MultivariateNormal(mus, covs)
    return kernels
'''

def softmax(logits):
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=1).reshape(logits.shape[0], 1)

def pred(logits):
    return torch.argmax(softmax(logits), axis=1)

# evaluate a model based on OVR ROC AUC and save confusion matrix with precision and recall
@torch.no_grad()
def evaluate(model, loader):
    y = []
    y_softmax = []
    y_pred = []

    for x_batch, y_batch in tqdm(loader):
        if use_cuda:
            x_batch = x_batch.cuda()
        y.extend(list(y_batch))
        logits, embeddings = model(x_batch)

        softmax_score = softmax(logits).detach().cpu().numpy()
        y_softmax.extend(softmax_score)
        y_pred.extend(np.argmax(softmax_score, axis=1))

    y = np.array(y)
    y_softmax = np.array(y_softmax)
    y_pred = np.array(y_pred)
    # ROC AUC score
    # if binary
    # fill the Nan by 1000
    if np.isnan(y_softmax).any():
        print("any data is nan",y_softmax)
        y_softmax=np.nan_to_num(y_softmax,nan=0.001)

    row_sums = np.sum(y_softmax, 1)
    row_sums = row_sums.repeat(y_softmax.shape[1])
    row_sums = row_sums.reshape(y_softmax.shape)
    y_softmax =y_softmax/row_sums 
    if class_num == 2:
        roc_auc = metrics.roc_auc_score(y, y_softmax[:, 1])
    else:
        roc_auc = metrics.roc_auc_score(y, y_softmax, multi_class='ovr')
    # confusion matrix, precision and recall
    cm = metrics.confusion_matrix(y, y_pred)
    true_class = np.sum(cm, axis=1)
    pred_class = np.sum(cm, axis=0)  
    precision = np.diag(cm) / pred_class
    recall = np.diag(cm) / true_class
    df = pd.DataFrame(cm)
    df['precision'] = precision
    df['recall'] = recall
    #df.to_csv(os.path.join(save_path, str(epoch)+'roc_auc'+str(roc_auc)+'.csv'))
    return roc_auc, precision, recall, df


def train(model, loader, optimizer, main_criterion, regu_criterion):
    running_loss = 0.
    batchs = 0
    for x_batch, y_batch in tqdm(loader):
        if use_cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        print(y_batch.max(),y_batch.min())
        optimizer.zero_grad()
        y_pred, embedding = model(x_batch)
        main_loss = main_criterion(y_pred, y_batch)
        if regu_criterion:
            regu_loss, _ = regu_criterion(embedding, y_batch)
            loss = main_loss + regu_loss
        else:
            loss = main_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batchs += 1

    return running_loss / batchs


def main(train_path,val_path,image_data_root, checkpoint_path=None):

    # define main and regulization loss functions
    main_criterion = nn.CrossEntropyLoss()
    regu_criterion = None #AMSoftmax(embedding_dim, class_num)
    # load data

    trainset = CustomDataset(train_path,image_data_root,transform=transform_train)
    valset = CustomDataset(val_path,image_data_root,transform=transform_test)

    trainloader = data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    valloader = data.DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        max_roc_auc = checkpoint['max_roc_auc']

        print()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('loaded checkpoint from path:', checkpoint_path, 'with start epoch:', start_epoch, 'loss:', loss, 'and max_roc_auc:', max_roc_auc)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print()

    else:
        max_score = -1
        start_epoch = 0

    # in distribution training? flag
    for i in tqdm(range(start_epoch, epochs)):
        # switch the main loss to FocalLoss after 10 epochs
        """if i == 10:
            main_criterion = FocalLoss(class_num=class_num)
        # switch the regulization loss to AMSoftmax after 30 epochs
        if i == 30:
            regu_criterion = AMSoftmax(embedding_dim, class_num)
        print('epoch:', i)"""
        

        l = train(model, trainloader, optimizer, main_criterion, regu_criterion)
            
        time.sleep(5)
        roc_auc, precision, recall, df = evaluate(model, valloader)
        score = (roc_auc + np.mean(precision) + np.mean(recall)) / 3
        print('score:', score)
        print(df)
        # only saves model with higher score
        if score > max_score:
            max_score = score
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        # save model checkpoint and confusion matrix every 5 epochs
        '''if (i+1) % 10 == 0:
            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': l,
                    'max_roc_auc': max_roc_auc
                }, os.path.join(save_path, 'epoch'+str(i)+'.pth')
            )'''
        if (i+1) % 5 == 0:
            df.to_csv(os.path.join(save_path, 'epoch:'+str(i)+'score:'+str(score)+'.csv'))     


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
    main(train_path,val_path,image_data_root)

    # evaluate best model on test set
    testset = CustomDataset(test_path,image_data_root,transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)

    #model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    model.eval()

    roc_auc, precision, recall, df = evaluate(model, testloader)
    score = (roc_auc + np.mean(precision) + np.mean(recall)) / 3

    print()
    print('---------------------------------------------------------------------------------------------------')
    print('model evaluation on test set')
    print('score:', score)
    print('confusion matrix:')
    print(df)
    print('---------------------------------------------------------------------------------------------------')
    print()

    df.to_csv(os.path.join(save_path, 'testset_'+'score:'+str(score)+'.csv'))





