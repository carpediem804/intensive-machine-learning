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
        self.net = Net().cuda() #우리가만든 net을 선언하고 cuda= gpu에서 동작하겠따는 의미임 
   
    # MNIST dataset
        self.train_data = Dataset(train=True, size = args.image_size , data_root = args.data_root)
        self.test_data = Dataset(train=False, size = args.image_size , data_root = args.data_root )

    # data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                               batch_size=args.batch_size, 
                                               shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data,
                                              batch_size=args.batch_size, 
                                              shuffle=True)
    
    # create loss operation and optimizer #classification임으로 아래거씀
        self.loss_op = nn.CrossEntropyLoss().cuda()
        self.optim   = torch.optim.SGD(self.net.parameters(), lr=args.lr) #nn.module의 파라미터를 다 가져오는것 =net.parameters()
        
       # mnist의 train함수와 같은게 들어가야함 testloader는 안만들어도 됨 
       # 사이즈에 args 사이즈 넣어야함 
       # train_data = Dataset(train=True 테스트는 false로 정의해야함 ,size=args.image_size, data_root=args.data_root) <-이런식으로해야함
        ########################################################################
        
        #                             END OF YOUR CODE
        ########################################################################

        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
       

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                ################################################################
                # TODO: [2] Flower 분류 - 2번
                # Epoch 학습 코드 작성하기
                ################################################################
                
                images = Variable(inputs[0]).cuda()
                labels = Variable(inputs[1]).cuda()
                self.optim.zero_grad() #이전의 gradient를 0으로 
                outputs = self.net(images) # forward를 자동으로 부른다  #  output의 크기 =   ############[batsh_size,10 ==스코어값 ] prdict size 
                self.loss = self.loss_op(outputs, labels)
                self.loss.backward()
                self.optim.step()

                 ################################################################
                #                             END OF YOUR CODE
                ################################################################

            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.test_data)
                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                    format(epoch+1, args.max_epochs, self.loss.data[0], train_acc, test_acc))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, data):
        self.net.eval() #################
        args = self.args
        loader = DataLoader(data,
                            batch_size=32,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        num_correct, num_total = 0, 0
       
            ####################################################################
            # TODO: [2] Flower 분류 - 2번
            # Evaluation 코드 작성하기
            ####################################################################
            
            
        for inputs in loader:
            images  = Variable(inputs[0]).cuda()
            labels  = inputs[1].cuda()
        
            outputs = self.net(images)
            _, preds = torch.max(outputs.data, 1)

            num_correct += (preds == labels).sum()
            num_total += labels.size(0)
        pass
            ####################################################################
            #                             END OF YOUR CODE
            ####################################################################

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
 

