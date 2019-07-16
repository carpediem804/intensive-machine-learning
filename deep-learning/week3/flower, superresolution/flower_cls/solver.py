import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):

        ########################################################################
        # TODO: [2] Flower 분류 - 2번
        # 네트워크, loss 함수, optimizer, dataset, data_loader 정의하기
        # 네트워크와 loss 함수는 cuda화 시켜야함
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                ################################################################
                # TODO: [2] Flower 분류 - 2번
                # Epoch 학습 코드 작성하기
                ################################################################
                pass
                ################################################################
                #                             END OF YOUR CODE
                ################################################################

            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.test_data)
                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                    format(epoch+1, args.max_epochs, loss.data[0], train_acc, test_acc))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        num_correct, num_total = 0, 0
        for inputs in loader:
            ####################################################################
            # TODO: [2] Flower 분류 - 2번
            # Evaluation 코드 작성하기
            ####################################################################
            pass
            ####################################################################
            #                             END OF YOUR CODE
            ####################################################################

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
