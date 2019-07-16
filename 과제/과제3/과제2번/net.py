import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ########################################################################
        # TODO: [2] Flower 분류 - 2번
        # CNN-MNIST 코드를 활용하여 자유롭게 CNN 코드를 작성한다.
        ########################################################################
        pass
        ########################################################################
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #3은 input channel개수 32는 output채널 개수
            nn.BatchNorm2d(32), # 아웃풋 channel이랑 맞춰줘야함 
            nn.ReLU(),
            nn.MaxPool2d(2) #64*64 - >32*32변환
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # 아웃풋 channel이랑 맞춰줘야함 
            nn.ReLU(),
            nn.MaxPool2d(2) #32*32 -> 16*16변환
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # 아웃풋 channel이랑 맞춰줘야함 
            nn.ReLU(),
            nn.MaxPool2d(2) #16*16->8*8변환
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # 아웃풋 channel이랑 맞춰줘야함 
            nn.ReLU(),
            nn.MaxPool2d(2) #8*8-> 4*4변환 아래의 linear안에 4*4 넣어야함 
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # 아웃풋 channel이랑 맞춰줘야함 
            nn.ReLU(),
            nn.MaxPool2d(2) #4*4-> 2*2변환 아래의 linear안에 4*4 넣어야함 
        )
        self.fc = nn.Linear(4*4*32, 10)#  #maxpool2d에서나온 4*4와 outchannel 32를 곱해서 넣어야함 # 아웃풋은 10으로 확정 
        
        #                             END OF YOUR CODE
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: [2] Flower 분류 - 2번
        # CNN-MNIST 코드를 활용하여 자유롭게 CNN 코드를 작성한다.
        ########################################################################
        pass
        ########################################################################
        conv1 = self.conv1(x) # 첫번째 layer통과
        conv2 = self.conv2(conv1) # 두번째 layer통과
        conv3 = self.conv3(conv2)#세번째 layer 통과
        conv4 = self.conv4(conv3)#네번째 layer 통과 
        #conv5 = self.conv5(conv4)#네번째 layer 통과 
        flat  = conv4.view(conv4.size(0), -1) #  conv2.size(0)에 batch사이즈가 들어가있음 
        out = self.fc(flat)
        #                             END OF YOUR CODE
        ########################################################################
        return out
