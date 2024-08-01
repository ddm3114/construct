import models.hpsv2 as hpsv2
import pandas as pd
from PIL import Image
from utlis import get_data,auc_score
from torch.nn import Identity
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class hps_v2_1(nn.Module):
    def __init__(self,model_path=''):
        super(hps_v2_1, self).__init__()
        self.text_work = False
        self.hps,self.tokenizer,self.process_val,self.process_train = hpsv2.initialize(hps_version="v2.1",text_work = self.text_work)
        self.fc = nn.Linear(1024,1)
        self.preprocess = self.process_train

    def forward(self,image,text=''):
        image_features = unwrap_model(self.hps).visual(image)
        scores =self.fc(image_features)

        return scores

class hps_v2(nn.Module):
    def __init__(self,model_path=''):
        super(hps_v2, self).__init__()
        self.text_work = False
        self.hps,self.tokenizer,self.process_val,self.process_train = hpsv2.initialize(hps_version="v2.0",text_work = self.text_work)
        self.fc = nn.Linear(1024,1)
        self.preprocess = self.process_train

    def forward(self,image,text = ''):
        image_features = unwrap_model(self.hps).visual(image)
        scores =self.fc(image_features)
    
        return scores

if __name__ == '__main__':
    device = 'cuda'
    imgs_path = 'images/bad1.png'
   
    model = hps_v2()
    model.to(device)
    print(model)
    image = model.preprocess(Image.open(imgs_path)).unsqueeze(0).to(device=device, non_blocking=True)
    print(image.shape)
    prompt = '1'
    text = model.tokenizer([prompt]).to(device=device, non_blocking=True)
    result = model(image,text)
    print(result)

    
    