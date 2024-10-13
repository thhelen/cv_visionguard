import torch 
import torchvision
import torch.nn as nn
from torchvision.models import swin_v2_s, Swin_V2_S_Weights


C2L ={0: 'Female',
 1: 'AgeOver60',
 2: 'Age18-60',
 3: 'AgeLess18',
 4: 'Front',
 5: 'Side',
 6: 'Back',
 7: 'Hat',
 8: 'Beard',
 9: 'HandBag',
 10: 'ShoulderBag',
 11: 'Backpack',
 12: 'HoldObjectsInFront',
 13: 'ShortSleeve',
 14: 'LongSleeve',
 15: 'UpperStride',
 16: 'UpperLogo',
 17: 'UpperPlaid',
 18: 'UpperSplice',
 19: 'LowerStripe',
 20: 'LowerPattern',
 21: 'LongCoat',
 22: 'Trousers',
 23: 'Shorts',
 24: 'Skirt&Dress',
 25: 'boots'}

L2C ={'Female': 0,
 'AgeOver60': 1,
 'Age18-60': 2,
 'AgeLess18': 3,
 'Front': 4,
 'Side': 5,
 'Back': 6,
 'Hat': 7,
 'Glasses': 8,
 'HandBag': 9,
 'ShoulderBag': 10,
 'Backpack': 11,
 'HoldObjectsInFront': 12,
 'ShortSleeve': 13,
 'LongSleeve': 14,
 'UpperStride': 15,
 'UpperLogo': 16,
 'UpperPlaid': 17,
 'UpperSplice': 18,
 'LowerStripe': 19,
 'LowerPattern': 20,
 'LongCoat': 21,
 'Trousers': 22,
 'Shorts': 23,
 'Skirt&Dress': 24,
 'boots': 25}


def get_backbone():
    backbone = swin_v2_s(Swin_V2_S_Weights.IMAGENET1K_V1)
    return backbone.features[:5]

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        # Linear layer for attribute prediction
        
        self.conv = nn.Conv2d(in_channels,in_channels,3,1,1)
        self.batchNorm = nn.BatchNorm2d(in_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(24192, out_channels)
        self.dropout = nn.Dropout2d(0.25)
    def forward(self, x):
        #inpuit -> bx18x14x384
        x = x.permute(0,3,1,2)
        x = self.dropout(self.relu(self.batchNorm(self.conv(x))))
        x = self.dropout(self.relu(self.batchNorm(self.conv(x))))
        x = self.maxpool(x)
        # Apply the linear layer
        b,_,_,_= x.shape
        x = x.reshape(b,-1)
        output = self.linear(x)
        
        return output# b x 512
    
class PersonAttributeHead(nn.Module):
    def __init__(self, in_channels, num_attributes=26):
        super(PersonAttributeHead, self).__init__()
        
        # Linear layer for attribute prediction
        self.linear = nn.Linear(in_channels,128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_channels, num_attributes)
        

    def forward(self, x):
        # Apply the linear layer
        # x = self.relu(self.linear(x))
        output = self.output(x)

        return output

class PersonReIDHead(nn.Module):
    def __init__(self, in_channels, emb=128):
        super(PersonReIDHead, self).__init__()
        
        # Linear layer for attribute prediction
        self.linear = nn.Linear(in_channels,256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, emb)

    def forward(self, x):
        # Apply the linear layer
        x = self.relu(self.linear(x))
        output = self.output(x)

        return output


class VisionGuard(nn.Module):
    def __init__(self,emb_size=128,num_attr=26):
        super(VisionGuard, self).__init__()
        
        self.backbone = get_backbone()
        # bx18x14x384
        self.decoder = Decoder(384,512)
        self.head_attr = PersonAttributeHead(512,num_attr)
        self.head_reid = PersonReIDHead(512,emb_size)
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def c2l(self,c):
        return C2L[c]

    def l2c(self,label):
        return L2C[label]

    def forward(self,x):
        x = self.backbone(x)
        x = self.decoder(x)
        
        return self.head_attr(x), self.head_reid(x)
    

