import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import scipy.io
from utils import PedestrianAttributeDataset,show_grid,get_device,get_attr_dataloader,get_accuracy,get_infinite_loader,get_infinite_zip_loader
from model import VisionGuard
import os 


#Write train for boith reid and attr and seprate all trains and vals for both
def train(model,dataloader,criterion,optimizer,scheduler=None,num_epochs=100,device="cpu",out_path="outputs",tf_logs="tf_logs"):
    
    writer = SummaryWriter(tf_logs)
    model = model.to(device)
    
    for epoch in range(num_epochs):
        #train
        model.train()
        loop = tqdm(dataloader["train"],desc=f' Training Epoch {epoch + 1}/{num_epochs}', unit='batch')
        curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        total_targets = []
        total_preds = []
        average_loss =0
        total_acc =0
        total_loss = 0
        for batch_idx,data in enumerate(loop):
            optimizer.zero_grad()
            images,labels = data
            images = images.to(device,dtype = torch.float)
            labels = labels.to(device,dtype = torch.float)
            
            out,_ = model(images)
            loss = criterion(out,labels)
            total_targets.append(labels.view(-1))
            total_preds.append(out.view(-1))
            out = out.view(-1).detach().cpu()
            labels = labels.view(-1).detach().cpu()
            
            
            acc = get_accuracy(labels,out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log loss to TensorBoard
            iteration = epoch * len(loop) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), iteration)
            writer.add_scalar('Train/Accuracy', acc, iteration)
            loop.set_postfix(loss = loss.item(),lr = curr_lr,accuracy = acc*100)
            if batch_idx ==len(loop)-1:
                average_loss = total_loss /len(loop)
                total_targets = torch.cat(total_targets,axis =0).view(-1).detach().cpu()
                total_preds = torch.cat(total_preds,axis =0).view(-1).detach().cpu()
                total_acc = get_accuracy(total_targets,total_preds)
                loop.set_postfix(lr = curr_lr,loss = loss.item(),avg_loss = average_loss,accuracy = acc*100,epoch_accuracy = total_acc*100)
                
        

        writer.add_scalar("Train/EpochLoss", average_loss, global_step=epoch)
        writer.add_scalar("Train/EpochAcc", total_acc, global_step=epoch)
        writer.add_scalar("Train/Lr", curr_lr, global_step=epoch)
        
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            loop = tqdm(dataloader["val"],desc=f' Validation', unit='image')
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
                loss = criterion(out,labels)
                
                loop.set_postfix(loss = loss.item())
                
                val_loss += loss.item()
                if batch_idx ==len(loop)-1:
                    average_loss = val_loss / len(loop)
                    total_targets = torch.cat(total_targets,axis=0).view(-1).detach().cpu()
                    total_preds = torch.cat(total_preds,axis =0).view(-1).detach().cpu()
                    total_acc = get_accuracy(total_targets,total_preds)
                    loop.set_postfix(loss = loss.item(),avg_loss = average_loss,val_accuracy = total_acc*100)
            
            writer.add_scalar("Val/Loss", average_loss, global_step=epoch)
            writer.add_scalar("Val/Acc", total_acc, global_step=epoch)
           
            print(f"Validation Loss: {average_loss:.4f} Accuracy:{total_acc:.4f}")
        if scheduler:
            scheduler.step()
                
            
       
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        torch.save(model.state_dict(),os.path.join(out_path,'swin_transformer_model.pth'))
    
    writer.close()


def evaluate(model,dataloader,criterion,weights,device="cpu",out_path="outputs",tf_logs="tf_logs"):
    print("Starting Evaluation")
    writer = SummaryWriter(tf_logs)
    
    model.load_state_dict(torch.load(weights))
    model = model.to(device)
    print("Model loaded")
    
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        loop = tqdm(dataloader["test"],desc=f' Testing', unit='image')
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
            loss = criterion(out,labels)
            
            loop.set_postfix(loss = loss.item(),accuracy = get_accuracy(labels.cpu(),out.cpu()))
            
            test_loss += loss.item()
            if batch_idx ==len(loop)-1:
                average_loss = test_loss / len(loop)
                total_targets = torch.cat(total_targets,axis=0).view(-1).detach().cpu()
                total_preds = torch.cat(total_preds,axis =0).view(-1).detach().cpu()
                total_acc = get_accuracy(total_targets,total_preds)
                loop.set_postfix(loss = loss.item(),avg_loss = average_loss,test_accuracy = total_acc*100)
        
        writer.add_scalar("Test/AttrLoss", average_loss, global_step=1)
        writer.add_scalar("Test/AttrAcc", total_acc, global_step=1)
    
        print(f"Test AttrLoss: {average_loss:.4f} Test AttrAccuracy:{total_acc:.4f}")
            
    writer.close()



if __name__=="__main__":
    train_dataloader = get_attr_dataloader(annotation_path="../../projectcv/pa-100k/annotation/annotation.mat",image_folder="../../projectcv/pa-100k/release_data/",split = "Train",batch_size =32)
    val_dataloader = get_attr_dataloader(annotation_path="../../projectcv/pa-100k/annotation/annotation.mat",image_folder="../../projectcv/pa-100k/release_data/",split = "Val",batch_size =1)
    dataloader ={"train":train_dataloader,
                "val":val_dataloader}
    model = VisionGuard()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5,weight_decay = 1e-8)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01,step_size_up=10,mode="exp_range",gamma=0.85,cycle_momentum=False)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
    scheduler = None
    train(model,dataloader,criterion,optimizer,scheduler=scheduler,device=get_device(),out_path="/home/jupyter/visionguard/output")
    