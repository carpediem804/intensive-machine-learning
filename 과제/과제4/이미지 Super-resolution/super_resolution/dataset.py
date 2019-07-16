import os
import glob
import numpy as np
import PIL
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *

class Dataset(data.Dataset): #토치에서 dataset 상속
    def __init__(self, scale, train, **kwargs): ##train이 true 로받으면 flower/train으로 false면 dirnmae = flowe/test로 
        super(Dataset, self).__init__()

        self.scale = scale #scale를 4로 받을 것이다
        self.size = kwargs.get("size", -1) # -1 stands for original resolution
        self.data_root = kwargs.get("data_root", "./data")
        
        self._prepare_dataset(self.size, self.data_root)

        if train:
            dirname = os.path.join(self.data_root, "flower/train")
        else:
            dirname = os.path.join(self.data_root, "flower/test")

        self.paths = glob.glob(os.path.join(dirname, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index): #이미지를불러오기
        hr_image = Image.open(self.paths[index])

        w, h = hr_image.size #이미지 w,h불러오기
        lr_image = hr_image.resize((int(w/self.scale), int(h/self.scale)), 
                                   PIL.Image.BICUBIC) #scale만큼 작게해서 lr image만들기

        return self.transform(hr_image), self.transform(lr_image)

    def __len__(self):
        return len(self.paths)

    def _prepare_dataset(self, size, data_root):
        check = os.path.join(data_root, "flower")
        if not os.path.isdir(check):
            download_and_convert(size, data_root)
