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
        train_iter, data_info = load_shakespeare(args.batch_size, args.bptt_len)
        self.vocab_size = data_info["vocab_size"]
        self.TEXT = data_info["TEXT"]

        self.net = Net(self.vocab_size, args.embed_dim,
                       args.hidden_dim, args.num_layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)

        self.net     = self.net.cuda()
        self.loss_fn = self.loss_fn.cuda()

        self.args = args
        self.train_iter = train_iter

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.cuda()
                y = inputs.target.cuda()

                loss = 0
                for i in range(X.size(0)):
                    hidden = hidden if i > 0 else None
                    out, hidden = self.net(X[i, :], hidden)

                    out = out.view(args.batch_size, -1)
                    loss += self.loss_fn(out, y[i, :])

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (epoch+1) % args.print_every == 0:
                text = self.sample(length=100)
                print("Epoch [{}/{}] loss: {:.3f}"
                    .format(epoch+1, args.max_epochs, loss.data[0]/args.bptt_len))
                print(text, "\n")
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def sample(self, length, prime="First"):
        self.net.eval()
        args = self.args

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
            _, argmax = torch.max(out, 1)

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
