import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import scipy.io
from utils import PedestrianAttributeDataset,show_grid,get_device,get_attr_dataloader,get_reid_dataloader,get_accuracy,get_infinite_loader,get_infinite_zip_loader
from model import VisionGuard
import os 


#Write train for boith reid and attr and seprate all trains and vals for both
def train(model,dataloader,criterion_attr,criterion_reid,optimizer,scheduler=None,num_epochs=100,device="cpu",out_path="outputs",tf_logs="tf_logs"):
    
    writer = SummaryWriter(tf_logs)
    model = model.to(device)
    
    for epoch in range(num_epochs):
        #train
        model.train()
        loop_len = 2000
        loop = tqdm(dataloader["train"],desc=f' Training Epoch {epoch + 1}/{num_epochs}', unit='batch',total=loop_len)
        curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        total_attr_targets = []
        total_attr_preds = []
        
        total_reid_targets = []
        total_reid_preds = []
        
        average_loss =0
        total_acc =0
        total_loss = 0
        total_reid =0
        for batch_idx,data in enumerate(loop):
            optimizer.zero_grad()
            batch_attr,batch_reid = data
            # attr training
            images,labels = batch_attr
            images = images.to(device,dtype = torch.float)
            labels = labels.to(device,dtype = torch.float)
            
            out,_ = model(images)
            loss_attr = criterion_attr(out,labels)
            total_attr_targets.append(labels.view(-1))
            total_attr_preds.append(out.view(-1))
            out = out.view(-1).detach().cpu()
            labels = labels.view(-1).detach().cpu()
            
            
            acc = get_accuracy(labels,out)
            
            # reid
            a,p,n = batch_reid
            a = a.to(device,dtype = torch.float)
            p = p.to(device,dtype = torch.float)
            n = n.to(device,dtype = torch.float)
            
            _,out_a = model(a)
            _,out_p = model(p)
            _,out_n = model(n)
            loss_reid = criterion_reid(out_a,out_p,out_n)
            
            loss = loss_attr+loss_reid 
            total_reid += loss_reid.item()            
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()

            # Log loss to TensorBoard
            iteration = epoch * loop_len + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), iteration)
            writer.add_scalar('Train/Accuracy', acc, iteration)
            loop.set_postfix(loss = loss.item(),lr = curr_lr,accuracy = acc*100,attr_loss =loss_attr.item(),reid_loss=loss_reid.item())
            if batch_idx ==loop_len-1:
               
                average_loss = total_loss /loop_len
                average_reid_loss = total_reid/loop_len
                total_attr_targets = torch.cat(total_attr_targets,axis =0).view(-1).detach().cpu()
                total_attr_preds = torch.cat(total_attr_preds,axis =0).view(-1).detach().cpu()
                total_acc = get_accuracy(total_attr_targets,total_attr_preds)
                loop.set_postfix(lr = curr_lr,loss = loss.item(),avg_loss = average_loss,accuracy = acc*100,epoch_accuracy = total_acc*100,attr_loss =loss_attr.item(),reid_loss=average_reid_loss)
                break
                
        

        writer.add_scalar("Train/EpochLoss", average_loss, global_step=epoch)
        writer.add_scalar("Train/EpochAcc", total_acc, global_step=epoch)
        writer.add_scalar("Train/Lr", curr_lr, global_step=epoch)
        
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            loop = tqdm(dataloader["val"][0],desc=f' Validation Attr', unit='image')
            total_targets = []
            total_preds = []
            total_acc =0
            for batch_idx,data in enumerate(loop):
                images,labels = data
                images = images.to(device,dtype = torch.float)
                labels = labels.to(device,dtype = torch.float)
            
                out,_ = model(images)
                
                total_targets.append(labels.view(-1))
                total_preds.append(out.view(-1))
                loss = criterion_attr(out,labels)
                
                loop.set_postfix(loss = loss.item())
                
                val_loss += loss.item()
                if batch_idx ==len(loop)-1:
                    average_loss = val_loss / len(loop)
                    total_targets = torch.cat(total_targets,axis=0).view(-1).detach().cpu()
                    total_preds = torch.cat(total_preds,axis =0).view(-1).detach().cpu()
                    total_acc = get_accuracy(total_targets,total_preds)
                    loop.set_postfix(loss = loss.item(),avg_loss = average_loss,val_accuracy = total_acc*100)
            
            writer.add_scalar("Val/AttrLoss", average_loss, global_step=epoch)
            writer.add_scalar("Val/AttrAcc", total_acc, global_step=epoch)
           
            # print(f"Validation Loss: {average_loss:.4f} Accuracy:{total_acc:.4f}")
            #Val Reid
            val_loss = 0.0
            loop = tqdm(dataloader["val"][1],desc=f' Validation ReID', unit='image')
            total_a = []
            total_p = []
            total_n = []
            # total_acc =0
            for batch_idx,data in enumerate(loop):
                a,p,n = data
                a = a.to(device,dtype = torch.float)
                p = p.to(device,dtype = torch.float)
                n = n.to(device,dtype = torch.float)
            
                _,out_a = model(a)
                _,out_p = model(p)
                _,out_n = model(n)
                
                total_a.append(out_a)
                total_p.append(out_p)
                total_n.append(out_n)
                loss = criterion_reid(out_a,out_p,out_n)
                
                loop.set_postfix(loss = loss.item())
                
                val_loss += loss.item()
                if batch_idx ==len(loop)-1:
                    average_loss = val_loss / len(loop)
                    total_a = torch.cat(total_a,axis=0).detach()
                    total_p = torch.cat(total_p,axis=0).detach()
                    total_n = torch.cat(total_n,axis=0).detach()
                    loss = criterion_reid(total_a,total_p,total_n)
                    loop.set_postfix(avg_loss = loss.item())
            
            writer.add_scalar("Val/ReidLoss", average_loss, global_step=epoch)
            
            
        if scheduler:
            scheduler.step()
                
            
       
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        torch.save(model.state_dict(),os.path.join(out_path,'swin_transformer_combined.pth'))
    
    writer.close()


if __name__=="__main__":
    train_attr = get_attr_dataloader(annotation_path="../../projectcv/pa-100k/annotation/annotation.mat",image_folder="../../projectcv/pa-100k/release_data/",split = "Train",batch_size =16)
    val_attr = get_attr_dataloader(annotation_path="../../projectcv/pa-100k/annotation/annotation.mat",image_folder="../../projectcv/pa-100k/release_data/",split = "Val",batch_size =1)
    
    reid_root_dir='../../projectcv/Market-1501'
    train_reid,val_reid = get_reid_dataloader(reid_root_dir,batch_size=8)
    
    train_attr = get_infinite_loader(train_attr)
    train_reid = get_infinite_loader(train_reid)
    train_dataloader = get_infinite_zip_loader(train_attr,train_reid)
    
    dataloader ={"train":train_dataloader,
                "val":[val_attr,val_reid]}
    model = VisionGuard()
    criterion_attr = nn.BCEWithLogitsLoss()
    criterion_reid = nn.TripletMarginWithDistanceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5,weight_decay = 1e-8)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01,step_size_up=10,mode="exp_range",gamma=0.85,cycle_momentum=False)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
    scheduler = None
    train(model,dataloader,criterion_attr,criterion_reid,optimizer,scheduler=scheduler,device=get_device(),out_path="/home/jupyter/visionguard/output")

