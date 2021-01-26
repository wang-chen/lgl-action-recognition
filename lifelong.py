
import os
import sys
import tqdm
import copy
import torch
import os.path
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from gat import GAT
from fgn import FGN
from ward import WARD
from nonlifelong import performance
from torch_util import count_parameters
from torch_util import Timer, EarlyStopScheduler


class Lifelong(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net, self.args = net, args
        self.register_buffer('inputs', torch.Tensor(0, 5, 5, 50))
        self.register_buffer('targets', torch.LongTensor(0))
        self.sample_viewed = 0
        self.memory_order = torch.LongTensor()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.1)

    def observe(self, inputs, targets, reply=True):
        self.net.train()
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            outputs = self.net.forward(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        self.sample(inputs, targets)
        if reply:
            L = torch.randperm(self.inputs.size(0))
            minibatches = [L[n:n+self.args.batch_size] for n in range(0, len(L), self.args.batch_size)]
            for index in minibatches:
                self.optimizer.zero_grad()
                inputs, targets = self.inputs[index], self.targets[index]
                outputs = self.net.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def uniform_sample(self, inputs, targets):
        self.inputs = torch.cat((self.inputs, inputs), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)

        if self.inputs.size(0) > self.args.memory_size:
            idx = torch.randperm(self.inputs.size(0))[:self.args.memory_size]
            self.inputs, self.targets = self.inputs[idx], self.targets[idx]

    @torch.no_grad()
    def sample(self, inputs, targets):
        self.sample_viewed += inputs.size(0)
        self.memory_order += inputs.size(0)# increase the order 

        self.targets = torch.cat((self.targets, targets), dim=0)
        self.inputs = torch.cat((self.inputs,inputs), dim = 0)
        self.memory_order = torch.cat((self.memory_order, torch.LongTensor(list(range(inputs.size()[0]-1,-1,-1)))), dim = 0)# for debug

        node_len = int(self.inputs.size(0))
        ext_memory = node_len - self.args.memory_size
        if ext_memory > 0:
            mask = torch.zeros(node_len, dtype = bool) # mask inputs order targets
            reserve = self.args.memory_size # reserved memrory to be stored
            seg = np.append(np.arange(0,self.sample_viewed,self.sample_viewed/ext_memory),self.sample_viewed)
            for i in range(len(seg)-2,-1,-1):
                left = self.memory_order.ge(np.ceil(seg[i]))*self.memory_order.lt(np.floor(seg[i+1]))
                leftindex = left.nonzero(as_tuple=False)
                if leftindex.size()[0] > reserve/(i+1): # the quote is not enough, need to be reduced
                    leftindex = leftindex[torch.randperm(leftindex.size()[0])[:int(reserve/(i+1))]] # reserve the quote
                    mask[leftindex] = True
                else:
                    mask[leftindex] = True # the quote is enough
                reserve -= leftindex.size()[0] # deducte the quote
            self.inputs = self.inputs[mask]
            self.targets = self.targets[mask]
            self.memory_order = self.memory_order[mask]



if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--model", type=str, default='FGN', help="FGN or GAT")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='accuracy/cora-lgl-test', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--duration", type=int, default=50, help="duration")
    parser.add_argument("--batch-size", type=int, default=100, help="minibatch size")
    parser.add_argument("--eval-iter", type=int, default=300, help="evaluation iteration")
    parser.add_argument("--jump", type=int, default=1, help="reply samples")
    parser.add_argument("--iteration", type=int, default=50, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=1000, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    Nets = {'fgn':FGN, 'gat':GAT}
    Net = Nets[args.model.lower()]

    test_data = WARD(root=args.data_root, duration=args.duration, train=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data = WARD(root=args.data_root, duration=args.duration, train=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    writter = SummaryWriter()
    lgl = Lifelong(Net(), args).to(args.device)
    print('Parameters: %d'%(count_parameters(lgl)))
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_loader)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        lgl.observe(inputs, targets)
        if (batch_idx+1) % args.eval_iter == 0:
            test_acc = performance(test_loader, lgl.net, args.device)
            print('Test Acc: %f'%(test_acc))
