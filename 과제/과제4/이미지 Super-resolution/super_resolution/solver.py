        
import os
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
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
        # TODO: [3] 이미지 Super-resolution 2번
        # 네트워크, loss 함수, optimizer, dataset, data_loader 정의하기
        # 네트워크와 loss 함수는 cuda화 시켜야함
        ########################################################################
        self.net = Net(4).cuda()
        
        self.train_data = Dataset(train=True,  scale = 4,size = args.image_size , data_root = args.data_root)
        self.test_data = Dataset(train=False,  scale = 4,size = args.image_size , data_root = args.data_root )

   
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                               batch_size=args.batch_size, 
                                               shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data,
                                              batch_size=args.batch_size, 
                                              shuffle=True)
        
        
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        self.args = args
        self.optim  = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.loss_op = nn.MSELoss()


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
                # TODO: [3] 이미지 Super-resolution 2번
                # Epoch 학습 코드 작성하기
                ################################################################
                pass
                low = Variable(inputs[1]).cuda() ## 저해상도
                high = Variable(inputs[0]).cuda() #고해상도
                
                self.optim.zero_grad()
                fit_images = self.net(low)
               
                self.loss = self.loss_op(fit_images, high).cuda()
                self.loss.backward()
                self.optim.step()


            
                ################################################################
                #                             END OF YOUR CODE
                ################################################################

            if (epoch+1) % args.print_every == 0:
                psnr = self.evaluate(epoch+1)
                print("Epoch [{}/{}] PSNR: {:.3f}".format(epoch+1, args.max_epochs, psnr))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, global_step):
        args = self.args
        loader = DataLoader(self.test_data,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        self.net.eval()
        mean_psnr = 0
        for step, inputs in enumerate(loader):
            image_hr = Variable(inputs[0], requires_grad=False).cuda()
            image_lr = Variable(inputs[1], requires_grad=False).cuda()

            image_sr = self.net(image_lr)

            image_hr = image_hr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
            image_lr = image_lr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
            image_sr = image_sr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()

            mean_psnr += psnr(image_hr, image_sr) / len(self.test_data)

            # save images
            hr_path = os.path.join(args.result_dir, "epoch_{}_{}_HR.jpg".format(global_step, step))
            lr_path = os.path.join(args.result_dir, "epoch_{}_{}_LR.jpg".format(global_step, step))
            sr_path = os.path.join(args.result_dir, "epoch_{}_{}_SR.jpg".format(global_step, step))
            misc.imsave(hr_path, image_hr)
            misc.imsave(lr_path, image_lr)
            misc.imsave(sr_path, image_sr)

        return mean_psnr

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)


def psnr(im1, im2): # 이미지 이미지는 accuracy가 없어서 픽셀간의 공통같은거해서 구함 높으면 높을수록 좋음
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out

    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
