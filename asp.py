import torch
import torch.nn as nn
from models.asp import clip
import models.asp.simple_inference as si
from PIL import Image
from utlis import get_data,auc_score
class asp(nn.Module):
    def __init__(self,model_path=''):
        super(asp, self).__init__()
        
        self.asp, self.preprocess = clip.load("ViT-L/14", device="cuda")
        self.model2 = si.initialize()
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self, x):
        image_features = self.asp(x)
    
        im_emb_arr = si.normalized(image_features.cpu().detach().numpy() )
        prediction = self.model2(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        prediction = self.sigmoid(prediction)
        return prediction


if __name__ == '__main__':
    model = asp()
    img_path = 'images/bad1.png'
    pil_image = Image.open(img_path)
    image = model.preprocess(pil_image).unsqueeze(0).to(device=model.device)
    result = model(image)
    print(model)
    print(result)