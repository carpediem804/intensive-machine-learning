import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
from utils import *

class Solver():
    def __init__(self, args):
        
        # load SST dataset
        train_iter, val_iter, test_iter, sst_info = load_sst(args.batch_size, args.max_vocab)
        vocab_size = sst_info["vocab_size"]
        num_class  = sst_info["num_class"]
        TEXT = sst_info["TEXT"]

        print("[!] vocab_size: {}, num_class: {}".format(vocab_size, num_class))

        self.net = Net(TEXT, 
                       args.hidden_dim,
                       args.num_layers, num_class)
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      args.lr, weight_decay=args.weight_decay) # embedding 학습 안하므로 빼줘야해서 이렇게 씀 lambda = 익명함수  P:파라미터 true 일때 reqgurie 이렇게 하고 아니면 냅두자  weight_decay= L2 
        self.loss_fn = nn.CrossEntropyLoss() #분류문제임으로 
        
        self.net     = self.net.cuda()
        self.loss_fn = self.loss_fn.cuda()
        ##net과 lossfn선언
        self.args = args
        self.train_iter = train_iter
        self.val_iter   = val_iter
        self.test_iter  = test_iter 
        ##torch text에 다 내장되어있음 
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self): #학습시키기
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.cuda() #트레인에서 text가져오기
                y = inputs.label.cuda() #트레인에서 label 가져오기

                pred_y = self.net(X) #예측된 y
                loss = self.loss_fn(pred_y, y) #로스function사용해서 loss구하기

                self.optim.zero_grad()#gradient초기화
                loss.backward()#backward하기
                self.optim.step()# 적용
            
            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_iter)
                val_acc   = self.evaluate(self.val_iter)
                print("Epoch [{}/{}] train_acc: {:.3f}, val_acc: {:.3f}"
                    .format(epoch+1, args.max_epochs, train_acc, val_acc))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, iters): #계산하기 
        args = self.args

        self.net.eval()
        num_correct, num_total = 0, 0
        for step, inputs in enumerate(iters):
            X = inputs.text.cuda()
            y = inputs.label.cuda()

            pred_y = self.net(X)
            _, pred_y = torch.max(pred_y.data, 1)

            num_correct += (pred_y == y.data).sum()
            num_total += y.size(0)

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
