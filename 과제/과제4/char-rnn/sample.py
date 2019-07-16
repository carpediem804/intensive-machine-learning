#네트워크 구성
import torch.nn.functional as F
from net import Net

import torchtext.datasets as datasets
import os
import argparse
from utils import *

import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#데이터셋

#샘플함수 이용 
class Solver():
    
    
    
    
    def __init__(self, args,prime):

        # load shakespeare dataset
        train_iter, data_info = load_shakespeare(args.batch_size, args.bptt_len) ##bptt_len = loss 계산해서 미분할때 속도때문에 짜르는거 갯수 
        self.vocab_size = data_info["vocab_size"]
        self.TEXT = data_info["TEXT"]
        #위에거 할필요없음
        
        
        self.net = Net(self.vocab_size, args.embed_dim,
                       args.hidden_dim, args.num_layers)
        self.net     = self.net.cuda() #gpu에서 돌림 
        self.args = args #메인에서 설정된 ir, epoch, layer갯수 등등 을 가져옴 
       
        
 
    def sample(self, length, prime): #length = 몇글자 생성할지 , prime 첫번째 글자(어디서부터할지)
        self.net.load_state_dict(torch.load('checkpoint/char-rnn_130.pth'))
        self.net.eval()
        args = self.args
            #처음에 글자가 주어지고 주어진 글자 마지막꺼의 output값이 다음 거의 input 값으로 들어가야함
        # 
        samples = list(prime)

        # convert prime string to torch.LongTensor type
        prime = self.TEXT.process(prime, device=0, train=False).cuda()

        # prepare the first hidden state
        for i in range(prime.size(1)):
            hidden = hidden if i > 0 else None
            _, hidden = self.net(prime[:, i], hidden)

        X = prime[:, -1]
        self.TEXT.sequential = False
        for i in range(length):
            out, hidden = self.net(X, hidden)

            # sample the maximum probability character
            _, argmax = torch.max(out, 1)# torch.max : 처음에는  max값, max값이 몇번째인지 -> 2개 return

            # preprocessing for next iteration
            out = self.TEXT.vocab.itos[argmax.data[0]]
            X = self.TEXT.numericalize([out], device=0, train=False).cuda()

            samples.append(out.replace("<eos>", "\n")) #sample에 out값의 list를 넣음 
    
    
    
    

        self.TEXT.sequential = True
        

        return "".join(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001) ##learning late 설정
    parser.add_argument("--batch_size", type=int, default=128) ## batch_size = 128
    parser.add_argument("--max_epochs", type=int, default=200) # epoch를 200으로 
    parser.add_argument("--bptt_len", type=int, default=30) ##30개씩 잘라서 미분할때 함 
    
    parser.add_argument("--embed_dim", type=int, default=300) # embedding할때 차원정해주기
    parser.add_argument("--hidden_dim", type=int, default=512) #hidden layer의 차원정해주기
    parser.add_argument("--num_layers", type=int, default=3) #레이어 갯수 =3
    prime = input("첫단어를입력하세요")
    aa = input("출력단어갯수를입력하세요")
    length = int(aa)
    args = parser.parse_args()
    solver = Solver(args,prime)
    print(solver.sample(length,prime), "\n\n")

if __name__ == "__main__":
    main()

