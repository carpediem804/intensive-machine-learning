import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from net import Net

def eval(net, loader):
    net.eval()
    num_correct, num_total = 0, 0
    for inputs in loader:
        ########################################################################
        # TODO: [1] MNIST 분류 - 1번
        # 2) Evaluation 함수 작성
        # Hint: num_correct 함수는 현재 배치에서 맞은 개수, num_total은 현재 배치의 개수를
        #       누적하여 계산
        ########################################################################
        pass
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

    return num_correct / num_total


def train(args):
    net = Net().cuda()
    # tensorboard writer for log summary
    writer = SummaryWriter()

    # MNIST dataset
    train_dataset = datasets.MNIST(root="./data/",
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root="./data/",
                                  train=False,
                                  transform=transforms.ToTensor())

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    ############################################################################
    # TODO: [1] MNIST 분류 - 1번
    # Loss 함수 및 optimizer 정의하기
    ############################################################################

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    for epoch in range(args.max_epochs):
        net.train()
        for step, inputs in enumerate(train_loader):
            ####################################################################
            # TODO: [1] MNIST 분류 - 1번
            # Epoch 학습 코드 작성하기
            ####################################################################
            pass
            ####################################################################
            #                             END OF YOUR CODE
            ####################################################################

        acc = eval(net, test_loader)

        ########################################################################
        # TODO: [1] MNIST 분류 - 1번
        # Tensorboard에 loss, test_accuracy 출력하는 코드 작성
        # Hint: 각각 한줄로 작성 가능 (총 2줄)
        ########################################################################
        pass
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        print("Epoch [{}/{}] loss: {:.5f} test acc: {:.3f}"
              .format(epoch+1, args.max_epochs, loss.data[0], acc))

    torch.save(net.state_dict(), "mnist-final.pth")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
