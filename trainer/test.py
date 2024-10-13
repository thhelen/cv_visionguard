from utils import get_device, get_attr_dataloader
from model import VisionGuard
from engine import evaluate
import os
import torch.nn as nn

if __name__ == "__main__":
    test_attr = get_attr_dataloader(annotation_path="../../projectcv/pa-100k/annotation/annotation.mat",image_folder="../../projectcv/pa-100k/release_data/",split = "Test",batch_size =1)
    
    model = VisionGuard()
    
    criterion = nn.BCEWithLogitsLoss()
    dataloader = {"test":test_attr}
    weights_path = "../output/swin_transformer_combined.pth"

    evaluate(model,dataloader,criterion,weights=weights_path,device=get_device(),out_path="/home/jupyter/visionguard/output/test",tf_logs="../output/test/tf_logs")