import torch
from torchvision.io import read_image
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights,convnext_small,ConvNeXt_Small_Weights,convnext_base,ConvNeXt_Base_Weights,convnext_large,ConvNeXt_Large_Weights
from torchvision.models import densenet121,DenseNet121_Weights,densenet161,DenseNet161_Weights,densenet169,DenseNet169_Weights,densenet201,DenseNet201_Weights
from torchvision.models import resnet18,ResNet18_Weights,resnet34,ResNet34_Weights,resnet50,ResNet50_Weights,resnet101,ResNet101_Weights,resnet152,ResNet152_Weights

from torchvision.models import swin_t,Swin_T_Weights
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import hps
import asp
from PIL import Image
import torch
import torch.nn as nn
from utlis import preprocess
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


class ClassifierHead(nn.Module):
    def __init__(self, in_features,hidden_dim = 1024, num_classes=1024):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=True)
        
        self._initialize_weights()

    def forward(self, x):

        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)






    
    
class MyModel(nn.Module):
    def __init__(self, pretrained = True,num_classes=1000):
        super(MyModel, self).__init__()
        self._initialize_weights()
        pass

    def forward(self, x):

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## 通用于调取外部模型，只需要将mymodel的名字传入即可
class baseModel(torch.nn.Module):
    def __init__(self,model_name,pretrained = True,train_backbone = False,hidden_dim = 1024,num_classes = 200):
        super(baseModel, self).__init__()
        self.model = self.create_model(model_name,pretrained)
        if train_backbone:
            for param in self.model.parameters():
                param.requires_grad = True
        else:

            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'head'):
            num_ftrs = self.model.head.in_features
            self.model.head = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif hasattr(self.model, 'classifier'):
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
    
    def create_model(self,model_name,pretrained):
        if model_name == 'ResNet18':
            if pretrained  == True:
                model = resnet18(weights = ResNet18_Weights.DEFAULT)
            else:
                model = resnet18()
            print('ResNet18 model loaded')

        elif model_name == 'ResNet34':
            if pretrained == True:
                model = resnet34(weights = ResNet34_Weights.DEFAULT)
            else:
                model = resnet34()
            print('ResNet34 model loaded')

        elif model_name == 'ResNet50':
            if pretrained == True:
                model = resnet50(weights = 'DEFAULT')
            else:
                model = resnet50()
            print('ResNet50 model loaded')

        elif model_name == 'Swin_T':
            if pretrained == True:
                model = swin_t(weights = Swin_T_Weights.DEFAULT)
            else:
                model = swin_t()
            print('Swin_T model loaded')
        elif model_name == 'hps-v2.1':
            model = hps.hps_v2_1()
            print('hps-v2.1 model loaded')
        elif model_name == 'hps-v2':
            model = hps.hps_v2()
            print('hps-v2 model loaded')
        elif model_name == 'asp':
            model = asp.asp()
            print('asp model loaded')
        elif model_name == 'MyModel':
            model = MyModel(pretrained=pretrained)
            print('Your Custom_Model loaded')
        else:
            raise ValueError('model not supported')
        
        return model
    

if __name__ == "__main__":
    # device = 'cuda'
    # model = baseModel(model_name='hps-v2.1',pretrained=True,train_backbone=False,hidden_dim=1024,num_classes=2)   
    # imgs_path = ['images/bad1.png','images/oops.png']
    # label = [1,2]
    # model.to(device)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # image,label = preprocess(model,imgs_path,label)
    # print(image.shape)
    # result = model(image)
    # print(result)

    device = 'cuda'
    model = baseModel(model_name='hps-v2.1',pretrained=True,train_backbone=True,hidden_dim=1024,num_classes=2) 
    imgs_path = ['images/bad1.png','images/oops.png']
    label = [1,0]
    model.to(device)
    print(model)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    image,label = preprocess(model,imgs_path,label)
    print(image.shape)
    result = model(image)
    print(result)