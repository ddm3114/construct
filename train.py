import torch
from PIL import Image
import tqdm

from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    device = torch.device('cuda')
from utlis import read_sample,transform,save_dict,load_dict,preprocess,load_weight
import json
import torch.optim.lr_scheduler as lr_scheduler
from get_optim import get_optim
from get_dataloader import get_dataloader
from get_model import get_model
from get_loss import get_loss
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def train(hyperparameters):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
    args = hyperparameters

    id = args['id']
    dataset = args['dataset']
    data_type = args['data_type']
    data_root = args['data_root']
    augment  = args['augment']
    epochs = args['epochs']
    model = args['model']
    model_name = model
    num_classes = args['num_classes']
    optim = args['optimizer']
    pretrained = args['pretrained']
    print('pretrained:',pretrained)
    lr_head = args['lr_head']
    weight_decay = args['weight_decay']
    sche = args['scheduler']
    step_size = args['step_size']
    gamma = args['gamma']
    eta_min = args['eta_min']
    train_backbone = args['train_backbone']
    batch_size =args['batch_size']
    lr_backbone = args['lr_backbone'] if 'lr_backbone' in args else None
    hidden_dim = args['hidden_dim'] if 'hidden_dim' in args else None
    AutoCast = args['autocast']
    loss_name = args['loss_f']
    patience_set = args['patience']
    patience = patience_set
    load_model = False
    save_model = False
    if 'load_dir' in args and args['load_dir']:
        load_model = True
   
    if 'save_dir' in args and args['save_dir']:
        save_model = True
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f'runs/{id}')
    #tensorboard --logdir=runs

    model = get_model(model= model,
                      pretrained=pretrained,
                      train_backbone=train_backbone,
                      hidden_dim=hidden_dim,
                      num_classes=num_classes)
    if load_model:
        model = load_weight(model,
                            config["load_dir"])
        
    model.to(device)

    if augment:
        train_dataloader,val_dataloader,augment_dataloader = get_dataloader(dataset = dataset,
                                                                            batch_size=batch_size,
                                                                            augment = augment,
                                                                            data_root=data_root,
                                                                            data_type=data_type)
    else:
        train_dataloader,val_dataloader = get_dataloader(dataset = dataset,
                                                         batch_size=batch_size,
                                                         augment = augment,
                                                         data_root=data_root,
                                                         data_type=data_type)

    optimizer = get_optim(model,
                          lr_head,
                          lr_backbone,
                          weight_decay,
                          train_backbone,
                          optim=optim)
    if sche == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size=step_size,
                                        gamma=gamma)
    elif sche == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                                                   T_max=epochs, 
                                                   eta_min=eta_min)
    criterion = get_loss(loss_name=loss_name)
    train_loss = []
    val_loss = []
    accuracy_list = []
    
    for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))
            # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),'-->grad_value:', torch.mean(parms.grad)) 

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        train_accuracy_epoch = []
        train_accuracy_list = []
        train_iter = 0
        print('Training train data')
        if epoch % step_size == 0:
            print(f'[Train]lr is {scheduler.get_last_lr()}')
        train_dataloader = tqdm.tqdm(train_dataloader,desc="Training epoch:{}".format(epoch),position=0,leave=False)
        for iter,sample in enumerate(train_dataloader):
            train_iter= iter
            # if isinstance(sample[0][0], str) and data_type == 'image':
            #     img,label = read_sample(model,sample,transform)
            # elif isinstance(sample[0],torch.Tensor) and data_type == 'image':
            #     img,label = sample
                
            # elif isinstance(sample[0][0],str) and data_type == 'text':
            #     img,label = process_text(sample,model_name)
            # else:
            #     raise TypeError(f"'{type(model).__name__}' object has wrong type either str or torch.Tensor'")
            img,label = preprocess(model,sample[0],sample[1])

            img = img.to(device)
            label = label.to(device)
            if loss_name == 'BCE': 
                label = label.unsqueeze(1)
                img = img.type(torch.float).to(device)
                label = label.type(torch.float).to(device)

            optimizer.zero_grad()
            if AutoCast:
                with autocast():
                    output = model(img)
                    loss = criterion(output, label)
                    

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
            else:
                
                output = model(img)

                loss = criterion(output, label)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()

            train_loss_epoch.append(loss.item())
            accuracy = (output.argmax(1) == label).sum().item()/len(label)
            train_accuracy_epoch.append(accuracy)
            
            sub_iter =20
            if iter % sub_iter == 0 and iter != 0:
                iter_loss = sum(train_loss_epoch[iter-sub_iter:iter])/sub_iter
                print(f'[Training]epoch:{epoch},iter:{iter},loss: {iter_loss}')

                # for name, parms in model.named_parameters():
                #     if parms.grad is None and parms.requires_grad:
                #         print(f"[warning]grad of {name} is None")

            del img
            del label
        
        if augment:
            lam = 0.5
            print('Training augmenting data')
            for iter,sample in enumerate(augment_dataloader):
                img,label = sample

                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = model(img)
                loss = lam*criterion(output, label)

                loss.backward()
                optimizer.step()
                train_loss_epoch.append(loss.item()/lam)
                accuracy = (output.argmax(1) == label.argmax(1)).sum().item()/label.shape[0]
                train_accuracy_epoch.append(accuracy)
                del img
                del label
                if iter % 100 == 0:
                    print(f'[Training auguemted data]epoch:{epoch},iter:{train_iter+iter},loss: {loss.item()/lam}') 

        print(f'[Train]epoch:{epoch},train loss: {sum(train_loss_epoch)/len(train_loss_epoch)}')
        epoch_loss = sum(train_loss_epoch)/len(train_loss_epoch)
        epoch_accuracy = sum(train_accuracy_epoch)/len(train_accuracy_epoch)

        train_loss.append(loss)
        train_accuracy_list.append(epoch_accuracy)
        scheduler.step()
        
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)

        
        print('\n')
        print('Validating')
        val_loss_epoch = []
        val_accuracy_epoch = []
        model.eval()
        for sample in val_dataloader:
            # if isinstance(sample[0][0], str) and data_type == 'image':
            #     img,label = read_sample(sample,transform)
            # elif isinstance(sample[0],torch.Tensor) and data_type == 'image':
            #     img,label = sample
                
            # elif isinstance(sample[0][0],str) and data_type == 'text':

            #     img,label = process_text(sample,model_name)
            # else:
            #     raise TypeError(f"'{type(model).__name__}' object has wrong type either str or torch.Tensor'")
            img,label = preprocess(model,sample[0],sample[1])
            
        
            
            img = img.to(device)
            label = label.to(device)
            if loss_name == 'BCE': 
                label = label.unsqueeze(1)
                img = img.type(torch.float).to(device)
                label = label.type(torch.float).to(device)
                
            if AutoCast:
                with autocast():
                    output = model(img)
                    loss = criterion(output, label)
            else:
                output = model(img)
                loss = criterion(output, label)

            accuracy = (output.argmax(1) == label).sum().item()/len(label)
            val_loss_epoch.append(loss.item())
            val_accuracy_epoch.append(accuracy)
            del img
            del label
            

        epoch_loss = sum(val_loss_epoch)/len(val_loss_epoch)
        epoch_accuracy = sum(val_accuracy_epoch)/len(val_accuracy_epoch)

       
       
        

        val_loss.append(epoch_loss)    
        

        writer.add_scalar('val Loss', epoch_loss, epoch)
        writer.add_scalar('val Accuracy', epoch_accuracy, epoch)
        
        print(f'[val]val loss: {sum(val_loss_epoch)/len(val_loss_epoch)}')
        print(f'[val]accuracy: {epoch_accuracy}')

        if epoch != 0:
            if epoch_accuracy > max(accuracy_list):
                save_dict(model,
                          hyperparameters,
                          epoch=epoch)
                print(f'[val]model saved at epoch:{epoch}')
            if epoch_accuracy <= max(accuracy_list):
                patience -= 1
                if patience == 0:
                    print(f'[val]early stopping at epoch:{epoch}')
                    accuracy_list.append(epoch_accuracy)
                    print('-'*50)
                    writer.close()
                    print("you can use '$ tensorboard --logdir=runs' to manage your model")
                    break
            else:
                patience = patience_set
        else:
            save_dict(model,
                      hyperparameters,
                      epoch=epoch)
            print(f'[val]model saved at epoch:{epoch}')

        accuracy_list.append(epoch_accuracy)
        print('-'*50)

    if save_model:
        save_dict(model,hyperparameters)
    del model

    writer.close()
    print("you can use '$ tensorboard --logdir=runs' to manage your model")
    return train_loss,val_loss,accuracy_list

if __name__ == '__main__':
    
    with open('config.json', 'r') as f:
        configs = json.load(f)

    for config in configs:
        train_loss,val_loss,accuracy_list = train(config)