import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
from utils import *

class Solver():
    def __init__(self, args):

        # load shakespeare dataset
        train_iter, data_info = load_shakespeare(args.batch_size, args.bptt_len) ##bptt_len = loss 계산해서 미분할때 속도때문에 짜르는거 갯수 
        self.vocab_size = data_info["vocab_size"]
        self.TEXT = data_info["TEXT"]
        #위에거 할필요없음
        
        
        self.net = Net(self.vocab_size, args.embed_dim,
                       args.hidden_dim, args.num_layers)
        self.loss_fn = nn.CrossEntropyLoss() #결측함수 
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr) #adam으로 최적화 

        self.net     = self.net.cuda() #gpu에서 돌림 
        self.loss_fn = self.loss_fn.cuda()#gpu에서 돌림

        self.args = args #메인에서 설정된 ir, epoch, layer갯수 등등 을 가져옴 
        self.train_iter = train_iter

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

    def fit(self): #훈련시키는것 
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train() #트레인시키기 
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.cuda()  #ex hello 테스트할 값
                y = inputs.target.cuda() #ex ello 나와야되는 결과값

                loss = 0 
                for i in range(X.size(0)): # h e l l o 
                    hidden = hidden if i > 0 else None #처음이아니면 전에 계산된 hidden을 가져옴 처음이면 hidden = 0 
                    out, hidden = self.net(X[i, :], hidden) #X[i, :] 한번에 하나의 step만,hidden을넣기 

                    out = out.view(args.batch_size, -1) # out의 원래사이즈 => [n=3,c=class개수]
                    loss += self.loss_fn(out, y[i, :]) # 각 하나하나 loss를 구해서 더함

                self.optim.zero_grad()#gradient를 계산하여 최적화 하기
                loss.backward()
                self.optim.step()#최적화 적용

            if (epoch+1) % args.print_every == 0:
                text = self.sample(length=100)
                print("Epoch [{}/{}] loss: {:.3f}"
                    .format(epoch+1, args.max_epochs, loss.data[0]/args.bptt_len))
                print(text, "\n")
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def sample(self, length, prime="First"): #length = 몇글자 생성할지 ,prime = 처음의 단어 
        self.net.eval()
        args = self.args

        samples = list(prime)

        # convert prime string to torch.LongTensor type
        prime = self.TEXT.process(prime, device=0, train=False).cuda()
#########위에는 전처리 ###########################################################
        
        # prepare the first hidden state  first라는 시작단어를 넣어서 test 시작하기 전 hidden state를 만듬 
        for i in range(prime.size(1)):
            hidden = hidden if i > 0 else None
            _, hidden = self.net(prime[:, i], hidden)

        X = prime[:, -1] # pirme의 끝글자 first에선 t  원래는 t를 넣엇을때 나온값을 넣어줘야함 잘못짜셧음 
        self.TEXT.sequential = False
        for i in range(length):
            out, hidden = self.net(X, hidden)

            # sample the maximum probability character
            _, argmax = torch.max(out, 1) #글자 예측 예측 예측 socre로 예측 예측 예측  

            # preprocessing for next iteration
            out = self.TEXT.vocab.itos[argmax.data[0]]
            X = self.TEXT.numericalize([out], device=0, train=False).cuda()

            samples.append(out.replace("<eos>", "\n"))

        self.TEXT.sequential = True

        return "".join(samples)


    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
