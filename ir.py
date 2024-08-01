from PIL import Image
import torch
import ImageReward as RM
import torch.nn as nn
class ir(nn.Module):
    def __init__(self,model_path=''):
        super(ir, self).__init__()
        self.ir = RM.load("ImageReward-v1.0")
    def forward(self,image,text=''):
        
        return outputs