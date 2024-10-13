from threading import Thread
import numpy as np
import cv2
import math
from ultralytics import YOLO
try:
    # Works when we're at the top lovel and we call main.py
    from ..trainer.model import VisionGuard
    from ..trainer.utils import *
except ImportError:
    # If we're not in the top level
    # And we're trying to call the file directly
    import sys
    # add the submodules to $PATH
    # sys.path[0] is the current file's path
    sys.path.append(sys.path[0] + '/..')
    from trainer.model import VisionGuard
    from trainer.utils import *


class VisionGuardProcessor:
    def __init__(self, model_path='/Users/shunya/Project/visionguard/ui/weights/swin_transformer_model_best.pth',yolo_weights='yolov8n.pt'):


        self.device = get_device()
        self.model = VisionGuard()
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.eval()
        self.augument = get_augumentations()
        self.normz = get_normalize_t()
        self.print_attr = False 
        self.model.to(self.device)
        self.yolo = YOLO(yolo_weights)
        self.yolo.fuse()
        self.frame_count = 0
        self.bboxes= None
        self.colorDict = {"maroon":(153,0,0),
                          "white":(255,255,255),
                          "green":(0,102,0)}


    
    def predict(self,input_x):
        with torch.inference_mode():
            input_x = Image.fromarray(input_x, 'RGB')
            input_x = self.normz(self.augument(input_x)).unsqueeze(0).to(self.device)
            attr, _ = self.model(input_x)
            attr = torch.sigmoid(attr).detach().cpu().numpy()
            classes = np.where(attr>=0.5)[1]
            labels = [(self.model.c2l(c),attr[0][c])for c in classes]
        return labels
    def process_yolo(self,frame):
        
        out = self.yolo.predict(frame,verbose=False)
        boxes = out[0].boxes
        classes = boxes.cls.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        person = np.where((conf >=0.65) & (classes ==0))[0]
        self.bboxes = boxes.data.detach().cpu().numpy()[person]
       

    def process(self,in_p = None,out_p=None):
        while True:
           
            ret,frame =  in_p.recv()
            if not ret:
                out_p.send((False,None))
                break
            if self.frame_count%15 == 0:
                self.process_yolo(frame)
            self.frame_count +=1               
            for box in self.bboxes:
                x,y,w,h,_,_ = box
                x,y,w,h = int(x),int(y),int(w),int(h)
                frame_vis = frame[y:y+h,np.max((x-50,1)):np.min((x+w+50,frame.shape[1]))]
                labels = self.predict(frame_vis)
                cv2.rectangle(frame, (x, y), ( w,   h), self.colorDict['green'], 2)
                for label,conf in labels:
                    y = y+50
                    frame = cv2.putText(frame,label , (x, (y+h)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorDict["white"], 4)
                    frame = cv2.putText(frame,label , (x, (y+h)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorDict["maroon"], 2)
            out_p.send((True, frame))


               
                # return labels
    