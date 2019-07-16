import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #저해상도 쭉 가다가 마지막에 합치는거 pulling은 쓰면안댐 이미지크기 키우는거 transfor 뭐시기뭐시기 함 
    def __init__(self, scale):
        super(Net, self).__init__()

        self.scale = scale

        ########################################################################
        # TODO: [3] 이미지 Super-resolution 2번
        ########################################################################
        #self.conv1 = nn.Conv2d(3,64,3,1,1)
        #self.t_conv1 = nn.ConvTranspose2d(64,64,2,2,0)
        #self.t_conv1 = nn.ConvTranspose2d(64,64,2,2,0)
        #self.conv2 = nn.Conv2d(64,3,3,1,1)
        #self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2)
         )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
             #nn.MaxPool2d(2)
         )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
             #nn.MaxPool2d(2)
         )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,3,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
             #nn.MaxPool2d(2)
         )
        self.transpose_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64,64,2,2,0),
            
         )
        self.transpose_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64,64,2,2,0),
           
         )
        
        
        
        
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: [3] 이미지 Super-resolution 2번
        ########################################################################
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        conv4 = self.transpose_conv1(conv3)
        out = self.transpose_conv2(conv4)
        out = self.conv4(out)
        
        
        #print (out)
        #out = self.transpose_conv1(out)
        #print(out.size)
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        return out
