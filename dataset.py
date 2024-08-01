
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import json
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100,CIFAR10
from torch.utils.data import DataLoader,Dataset
import random
from collections import defaultdict
from typing import List, Dict
import pandas
import csv

class Augmentation_Dataset:
    def __init__(self,dataset,transform=None):
        self.transform = transform
        self.dataset = dataset
        self.length = int(dataset.__len__()//2)
        
        if dataset.classes:
            self.num_classes = len(dataset.classes)
        elif dataset.num_classes:
            self.num_classes = dataset.num_classes
        else:
            raise ValueError('Number of classes not found')
        self.cutmix = T.CutMix(num_classes = self.num_classes)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        img,label = self.augmentation(idx)
        
        
        
        return img, label
    
    def augmentation(self,idx):
        imgs = []
        labels = []

        image, label = self.dataset.__getitem__(2*idx)
        imgs.append(image)
        labels.append(label)
        image, label = self.dataset.__getitem__(2*idx+1)
        imgs.append(image)
        labels.append(label)


        # to_tensor = ToTensor()
       
        labels = torch.tensor(labels)
        imgs = torch.stack(imgs)
        img, label = self.cutmix(imgs, labels)
        return img[0],label[0]
                  

# 加载CIFAR-100数据集
class CIFAR100_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR100(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR100 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
    
   
class CIFAR10_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR10 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes

class Custom_Dataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label
        self.length = len(data)
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

class My_Dataset:
    def __init__(self,data_root):
        self.data_root = data_root
        datas = pandas.read_csv(data_root)
        datas = datas.sample(frac=1).reset_index(drop=True)

        self.length = len(datas)
        data = datas['img_path'].tolist()
        label = datas['label'].tolist()
        train_split = int(0.85*len(data))
        val_split = len(data)
        test_split = len(data)
        self.train_dataset = Custom_Dataset(data[:train_split],label[:train_split])
        self.val_dataset = Custom_Dataset(data[train_split:val_split],label[train_split:val_split])
        # self.test_dataset = Custom_Dataset(data[val_split:test_split],label[val_split:test_split])



    
if __name__ == '__main__':
    # dataset = CIFAR100_Dataset(transform=transform)
    # train_dataset,test_dataset = dataset.train_dataset,dataset.test_dataset
    # num_classes = dataset.num_classes
    # augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # augmented_dataloader = DataLoader(augmented_dataset,batch_size=2,shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # for i, (img, label) in enumerate(augmented_dataloader):
    #     print(img.shape,label.shape)
        
    #     img_pth = f'result/image_{i}.png'
        
    #     show_image(img[0],img_pth)
    #     print("label:",label[0])
    #     break
    data_root = 'data/data_label.csv'
    dataset = My_Dataset(data_root)
    train_dataset = dataset.train_dataset
    print(train_dataset.__getitem__(2))
    print(len(train_dataset))
    print(train_dataset.label[:20])
    val_dataset = dataset.val_dataset
    print(len(val_dataset))
    test_dataset = dataset.test_dataset
    print(len(test_dataset))