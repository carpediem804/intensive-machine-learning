import os
import csv
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *

class Dataset(data.Dataset): #토치에서 dataset상속
    str2label = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}
    label2str = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
    #숫자로 바꿀라고 static으로 선언함  숫자와 클래스 연결 
    
    #data = Dataset(train=Flase, size = 100, data_root = 'mydata'
    
    def __init__(self, train, **kwargs): # ** **kwargs가 data를 딕셔너리로 만들어줌  tarin을 true로받으면 train.csv로만듬
        super(Dataset, self).__init__()
    #d.get은 2개를 리턴 
        self.data = list()
        self.size = kwargs.get("size", -1) # -1 stands for the original resolution
        self.data_root = kwargs.get("data_root", "./data")
        
        self._prepare_dataset(self.size, self.data_root)
        
        #train이나 test csv에 data_path, daisy 이미지위치, 이미지라벨이 있음  
        #data_path2, dandelion
        csv_name = "train.csv" if train else "test.csv"  # train이 true일때 train.csv만들어주고 false로받으면 test.csv를 만들어줌 
        with open(os.path.join(self.data_root, "flower", csv_name)) as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

        if train: #train이 true일 때 data를 섞음
            random.shuffle(self.data)

        self.transform = transforms.Compose([
            transforms.ToTensor() # 텐서타입으로 바꾸는거 
        ])

    def __getitem__(self, index): #이미지를 실제로 불러오는거 
        path, label = self.data[index]
        image = Image.open(path) # 이미지를 불러옴
        label = self.str2label[label] #daisy는 0 이런거

        return self.transform(image), label # 이미지는 텐서타입 라벨은 정수 로 리턴 

    def __len__(self):
        return len(self.data)
##밑에거 주석달지말기 이거랑 util.py는 주석달지말기 
    def _prepare_dataset(self, size, data_root):
        check = os.path.join(data_root, "flower")
        download_and_convert(size, data_root)
        if not os.path.isdir(check):
            download_and_convert(size, data_root)
