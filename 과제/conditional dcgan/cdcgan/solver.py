import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from net import *

class Solver():
    def __init__(self, args):
        # define normalize transformation
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))
        ])

        # load the fashion MNIST dataset
        self.train_dataset = datasets.FashionMNIST(
            root=args.data_root,
            train=True, 
            transform=transform, 
            download=True)
 
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        
        self.G = Generator(z_dim=args.z_dim)
        self.D = Discriminator()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim_G = torch.optim.Adam(self.G.parameters(), args.lr)
        self.optim_D = torch.optim.Adam(self.D.parameters(), args.lr)

        # cudafy objects
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        self.args = args
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.G.train()
            self.D.train()
            for step, inputs in enumerate(self.train_loader):
                batch_size = inputs[0].size(0)
                image = Variable(inputs[0], requires_grad=False).cuda()
                label = Variable(inputs[1], requires_grad=False).cuda()

                # create the labels used to distingush real or fake
                real_labels = Variable(torch.ones(batch_size).type(torch.LongTensor)).cuda()
                fake_labels = Variable(torch.zeros(batch_size).type(torch.LongTensor)).cuda()
                
                # train the discriminator  
                D_real, D_real_cls = self.D(image)
                D_loss_real = self.loss_fn(D_real, real_labels)
                D_loss_real_cls = self.loss_fn(D_real_cls, label)
                
                z = Variable(torch.randn(batch_size, args.z_dim)).cuda()

                # make label to onehot vector
                y_onehot = torch.FloatTensor(batch_size, 10)
                y_onehot.zero_()
                y_onehot.scatter_(1, inputs[1].unsqueeze(1), 1)
                y_onehot = Variable(y_onehot, requires_grad=False).cuda()
                
                G_fake = self.G(y_onehot, z)
                D_fake, D_fake_cls = self.D(G_fake)
                D_loss_fake = self.loss_fn(D_fake, fake_labels)
                D_loss_fake_cls = self.loss_fn(D_fake_cls, label)
                
                D_loss = D_loss_real + D_loss_fake + \
                         D_loss_real_cls + D_loss_fake_cls
                self.D.zero_grad()
                D_loss.backward()
                self.optim_D.step()
                
                # train the generator
                z = Variable(torch.randn(batch_size, args.z_dim)).cuda()
                G_fake = self.G(y_onehot, z)
                D_fake, D_fake_cls = self.D(G_fake)
                
                G_loss = self.loss_fn(D_fake, real_labels) + \
                         self.loss_fn(D_fake_cls, label)
                self.G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}] Loss_D: {:.3f}, Loss_G: {:.3f}".
                    format(epoch+1, args.max_epochs, D_loss.data[0], G_loss.data[0]))
                self.save(args.ckpt_dir, epoch+1)
                self.sample(epoch+1)

                
    def sample(self, global_step=0):
        self.G.eval()
        self.D.eval()
        
        args = self.args
        batch_size = args.batch_size
                
        # produce the samples among 10-classes
        for i in range(10):
            z = Variable(torch.randn(batch_size, args.z_dim),
                     volatile=True).cuda()
            y = torch.LongTensor(batch_size).fill_(i)
            
            # make label to onehot vector
            y_onehot = torch.FloatTensor(batch_size, 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y_onehot = Variable(y_onehot, volatile=True).cuda()

            G_fake = self.G(y_onehot, z)

            # save the results
            save_image(denormalize(G_fake.data),
                os.path.join(args.result_dir, "fake_{}_{}.png".format(global_step, i)))


    def save(self, ckpt_dir, global_step):
        D_path = os.path.join(
            ckpt_dir, "discriminator_{}.pth".format(global_step))
        G_path = os.path.join(
            ckpt_dir, "generator_{}.pth".format(global_step))

        torch.save(self.D.state_dict(), D_path)
        torch.save(self.G.state_dict(), G_path)


def denormalize(tensor):
    out = (tensor + 1) / 2
    return out.clamp(0, 1)

