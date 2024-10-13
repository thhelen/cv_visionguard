import torch 
import torchvision
from torch.utils.data import Dataset,random_split,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as v2
import torchvision.transforms.functional as F

from collections import defaultdict
import random

import matplotlib.pyplot as plt

from PIL import Image
import scipy.io
import numpy as np
import os
import itertools


from sklearn.metrics import accuracy_score

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

np.random.seed(0)
torch.manual_seed(0)

def load_image(path:str):
    image = Image.open(path)
    if image.mode !='RGB' or image.mode != 'RGBA': # converts grayscale image to RGB
        image = image.convert("RGB")
    return image

def get_augumentations():
    return v2.Compose([
        v2.Resize([256,192]),
        v2.Pad(10),
        # v2.RandomResizedCrop([256,192]),
        ]) 
def get_normalize_t():
    return v2.Compose([v2.ToTensor(),v2.Normalize(IMG_MEAN, IMG_STD)])

class PedestrianAttributeDataset(Dataset):
    
    def __init__(self,annotation_path:str,image_folder:str,split = "Train"):
        self.annotations =scipy.io.loadmat(annotation_path)
        self.image_folder = image_folder
        self.split = split
        self.files, self.labels = self.get_files_labels()
        # self.files = self.annotations["train_images_name"]
        # self.labels = self.annotations["train_label"]
        self.classes = self.annotations["attributes"]
        self.class2label = {x[0][0]:i for i,x in enumerate(self.classes)}
        self.label2class = {i:x[0][0] for i,x in enumerate(self.classes)}
        self.augmentations = get_augumentations()
        self.train_transforms = v2.Compose([ v2.RandomRotation(5),
        v2.RandomHorizontalFlip(0.5)])
        self.normalize = get_normalize_t()
    
    def get_files_labels(self):
        match self.split:
            
            case "Train":
                return self.annotations["train_images_name"],self.annotations["train_label"]
            case "Val":
                return self.annotations["val_images_name"],self.annotations["val_label"]
            case "Test":
                return self.annotations["test_images_name"],self.annotations["test_label"]
            case _ :
                raise NotFoundError(f"{self.split} not found.")

            
            
    def __getitem__(self,index):
        img = load_image(os.path.join(self.image_folder,self.files[index,0][0]))
        
        img = self.augmentations(img)
        if self.split == "Train":
            img = self.train_transforms(img)
        return self.normalize(img), self.labels[index]
        

    def __len__(self):
        return len(self.files)
    
    def c2l(self,x:str):
        return self.class2label[x]
    
    def l2c(self,x:int):
        return self.label2class[x]


def get_classes_from_lables(annotation:str,i):
    annotations =scipy.io.loadmat(annotation)
    classes = annotations["attributes"]
    class2label = {x[0][0]:i for i,x in enumerate(classes)}
    return class2label[i]
    
    
class PedestrianReIDDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir,"bounding_box_train")
        self.files_list = os.listdir(self.image_folder)
        self.files = self.get_file_dict()
        self.labels = list(self.files.keys())
        self.train_transforms = v2.Compose([ v2.RandomRotation(5),
        v2.RandomHorizontalFlip(0.5)])
        self.augmentations = get_augumentations()
        self.normalize = get_normalize_t()
        
    def get_file_dict(self):
        c = defaultdict(list)
        for file in self.files_list:
            x = file.split("_")[0]
            c[x].append(file)
        return c
            
    def __getitem__(self, index):
        anchor_name,negative_name = random.choices(self.labels, k=2)
        anchor_name,positive_name = random.choices(self.files[anchor_name],k=2)
        anchor = load_image(os.path.join(self.image_folder,anchor_name))
        positive = load_image(os.path.join(self.image_folder,positive_name))

        # Select a negative sample (image with a different label)
        negative_name = random.choice(self.files[negative_name])
        
        negative  = load_image(os.path.join(self.image_folder,negative_name))
        
        anchor, positive,negative = self.augmentations(anchor),self.augmentations(positive),self.augmentations(negative)
        if self.train_transforms:
            anchor = self.train_transforms(anchor)
            positive = self.train_transforms(positive)
            negative = self.train_transforms(negative)

        return self.normalize(anchor), self.normalize(positive), self.normalize(negative)

    def __len__(self):
        return len(self.labels)


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1)

def show_grid(imgs):
    if not isinstance(imgs, list):
        imgs = denormalize(imgs)
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=(15, 10))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_device():

    # if not torch.backends.mps.is_available():
    #     if not torch.backends.mps.is_built():
    #         print("MPS not available because the current PyTorch install was not "
    #           "built with MPS enabled.")
    #     else:
    #         print("MPS not available because the current MacOS version is not 12.3+ "
    #           "and/or you do not have an MPS-enabled device on this machine.")
            
    #     device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # else:
    #     print("Using MPS")
    #     device = torch.device("mps")
    # return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attr_dataloader(annotation_path="../gcs/pa-100k/annotation/annotation.mat",image_folder="../gcs/pa-100k/release_data/",split = "Train",batch_size=4):
    match split:
        case "Train":
            train_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            l = len(train_dataset)//2
            train_dataset,_ = random_split(train_dataset,(l,l))
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        case "Val":
            val_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            l = len(val_dataset)//2
            val_dataset,_ = random_split(val_dataset,(l,l))
            return DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        case "Test":
            test_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            return test_dataset,DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
def get_reid_dataloader(root_dir,split="TrainVal",batch_size=4,validation_split = .1):
    train_dataset = PedestrianReIDDataset(root_dir)
    match split:
        case "Train":
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        case "TrainVal":
            random_seed= 42
            dataset_size = len(train_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,sampler = train_sampler)
            val_loader = DataLoader(train_dataset, batch_size=1,drop_last=True,sampler = valid_sampler)
            return train_dataloader,val_loader
        
        case "Test":
            test_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

            

def get_accuracy(y_true,y_pred):
    y_pred = torch.sigmoid(y_pred)
    y_pred[y_pred >=0.5] =1
    y_pred[y_pred<0.5] =0
    
    return accuracy_score(y_true,y_pred)



def get_infinite_loader(loader1):
    iterator1 = iter(loader1)

    while True:
        try:
            batch1 = next(iterator1)
            yield batch1
        except StopIteration:
            # Reset iterator1 when it reaches the end
            iterator1 = iter(loader1)

def get_infinite_zip_loader(loader1, loader2):
    iterator1 = iter(loader1)
    iterator2 = iter(loader2)

    while True:
        batch1 = next(iterator1)
        batch2 = next(iterator2)
        yield batch1, batch2

