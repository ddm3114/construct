import torch
from dataset import My_Dataset
from torch.utils.data import DataLoader
from utlis import load_dict,process_text,load_weight
from get_model import get_model
import os
if torch.cuda.is_available():
    device = torch.device('cuda')
import torch.nn as nn
import json
import csv
from PIL import Image
from tqdm import tqdm
def inference(config,batch=10,data_root = None):
    device = "cuda"
    output_path = "output_case5.csv"
    if not data_root:
        data_root = config['data_root']

    load_path = os.path.join(config['save_dir'],'model_13.pth')
    # dataset = My_Dataset(data_root)
    # label_list = dataset.label_list
    # test_dataset = dataset.test_dataset
    # test_dataloader = DataLoader(test_dataset,batch_size=batch,shuffle=True)
    model_name = config['model']
    model = get_model(model_name,
                      pretrained=config['pretrained'],
                      num_classes=config['num_classes'],
                      hidden_dim=config['hidden_dim'])
    model = load_weight(model,load_path)

    model.to(device)
    
    model.eval()

    bad_case  = 0
    file_names = os.listdir(data_root)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'label'])
    
        for file in tqdm(file_names):
            file_path = os.path.join(data_root, file)
            img = Image.open(file_path)
            img = model.model.process_val(img)
            img = img.unsqueeze(0)
            img = img.to(device, non_blocking=True)
            output = model(img)
            label = torch.argmax(output, dim=1).item()
            
            if label == 1:
                writer.writerow([file_path, label])
                bad_case += 1
    print(f"Bad case: {bad_case}/{len(file_names)}")


        

if __name__ == '__main__':
    path =['hps-2.1pth/v2/config.json']
    data_root = 'data/data5'
    for p in path:
        with open(p) as f:
            config = json.load(f)
        inference(config,data_root=data_root)