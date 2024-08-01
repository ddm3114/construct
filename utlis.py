
import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import json


from transformers import BertTokenizer
import numpy as np

from sklearn.metrics import roc_curve, auc
import pandas as pd
from tqdm import tqdm
import requests
import os
from PIL import Image
from io import BytesIO


if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def preprocess(model,imgs,label,text = '',non_blocking = True):
    images = []
    for img in imgs:
        image = model.model.preprocess(Image.open(img)).to(device=device, non_blocking=non_blocking)
        images.append(image)
    images = torch.stack(images)
    if text != '':
        text = model.model.tokenizer(text).to(device=device, non_blocking=non_blocking)

    label = torch.tensor(label).to(device=device)

    return images,label



def read_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ToTensor()(img)
    return img

def show_image(img,save_path):
    img.squeeze_(0)
    print('the shape of the image is :',img.shape)
    to_pil = ToPILImage()
    img_pil = to_pil(img)
    plt.imshow(img_pil)
    img_pil.save(save_path)
    print(f"Image saved to {save_path}")

def read_sample(sample,transform =None):
    imgs =[]
    labels = []
    for i in range(len(sample[0])):
        img = sample[0][i]
        label = sample[1][i]
    
        img = Image.open(img).convert('RGB')
        if transform:
            img = transform(img)
        else:
            img = ToTensor()(img)
        if img.shape[0] != 3:
            continue
        if label >=200:
            continue
        
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    return imgs, labels

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用常用的均值和标准差进行标准化
])



def save_dict(model,config,**args):
    os.makedirs(config['save_dir'],exist_ok=True)
    if 'epoch' in args:
        epoch = args['epoch']
        save_path = os.path.join(config['save_dir'],f'model_{epoch}.pth')
        torch.save(model.state_dict(),save_path)
        print(f'Model saved to {save_path}')
        return
    save_path = os.path.join(config['save_dir'],'model.pth')
    torch.save(model.state_dict(),save_path)

    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'Model and config saved to {config["save_dir"]}')

def load_dict(model,config):
    load_path = os.path.join(config['load_dir'],'model.pth')
    pretrained_dict = torch.load(load_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc' and k != 'head' and k != 'classifier'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict) 
    
    print(f'model loaded from {load_path}')

    return model

train_transforms = transforms.Compose([
    transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR),
    ToPILImage(),
    transforms.RandomResizedCrop(size=64),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


# def process_text(sample,model_name):
#     text,label = sample
#     if model_name == 'Bert':
#         text = bert.process(text)
#     elif model_name == 'RoBerta':
#         text = roberta.process(text)
#     else:
#         raise ValueError(f"model_name should be either 'Bert' or 'RoBerta' but got {model_name}")
    
#     if not isinstance(label,torch.Tensor):
#         label = torch.tensor(label)
#     return text,label

def load_weight(model,weight_path):
    pretrained_dict = torch.load(weight_path)
    model.load_state_dict(pretrained_dict)
    print(f'model loaded from {weight_path}')
    return model


def get_data(data_path):
    data = pd.read_csv(data_path)
    imgs_path = data['image_path'].tolist()
    prompt = data['prompt'].tolist()
    label = data['bad_case'].tolist()
    return imgs_path,prompt,label

def auc_score(true_label, result_label,step = 0.0005):
    left = min(result_label)
    right = max(result_label)
    thresholds = np.arange(left,right, step)  # 从0到1，步长为0.05

    # 初始化假阳性率、真阳性率和AUC
    fpr_list = []
    tpr_list = []

    for threshold in thresholds:
        # 将预测分数转换为二进制标签
        binary_predictions = [score >= threshold for score in result_label]
        
        # 计算TPR和FPR
        tp = sum((pred == True and label == True) for pred, label in zip(binary_predictions, true_label))
        fp = sum((pred == True and label == False) for pred, label in zip(binary_predictions, true_label))
        fn = sum((pred == False and label == True) for pred, label in zip(binary_predictions, true_label))
        tn = sum((pred == False and label == False) for pred, label in zip(binary_predictions, true_label))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # 计算AUC
    roc_auc = auc(fpr_list, tpr_list)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    plt.show()

    # 寻找最佳阈值
    optimal_idx = np.argmax(np.array(tpr_list) - np.array(fpr_list))
    optimal_threshold = thresholds[optimal_idx]
    print(f'最佳阈值: {optimal_threshold}')
    return roc_auc, optimal_threshold

oops_image = Image.open("images/oops.png")

def filter_images(img):
    return oops_image.tobytes() == img.tobytes()

def get_image_from_url(root_path = 'data/data.csv', download_directory = 'bad_case', output_csv_file_path = 'bad_case/case.csv'):
    
    oops_num = 0
    df = pd.read_csv(root_path)
    urls = df['photo_url'].tolist()
    prompts = df['prompt'].tolist()
    images_num = len(urls)
    downloaded_images_info = []
    for url, prompt in tqdm(zip(urls, prompts), total=len(urls)):
        try:
            image_name = os.path.basename(url)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                downloaded_image = Image.open(BytesIO(response.content))
                if not filter_images(downloaded_image):
                    image_path = os.path.join(download_directory, image_name)
                    downloaded_image.save(image_path)
                    downloaded_images_info.append({'image_path': image_path, 'prompt': prompt})
                else:
                    oops_num += 1
                    print(f"Skipped {url} as it is identical to ooops.png")
            else:
                print(f"Failed to download {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    downloaded_images_df = pd.DataFrame(downloaded_images_info)
    downloaded_images_df.to_csv(output_csv_file_path, index=False)
    print(f"all images: {images_num}, oops images: {oops_num},downloaded images: {images_num - oops_num}")