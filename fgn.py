import torch
import numpy as np
import torch.nn as nn


class FGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat1 = Trans1d(5, 32, kernel_size=(5,11), stride=1, padding=(0,5))
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.MaxPool2d((1,2)), nn.ReLU())
        self.feat2 = Trans1d(32, 32, kernel_size=(1,5), stride=1, padding=(0,2))
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.MaxPool2d((1,2)), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32*12, 13))

    def forward(self, x):
        assert x.size(-1)%2 == 0
        x, n = x.split([x.size(-1)//2, x.size(-1)//2], dim=-1)
        x, n = self.feat1(x, x), self.feat1(n, n)
        x, n = self.acvt1(x), self.acvt1(n)
        x, n = self.feat2(x, n), self.feat2(n, x)
        x, n = self.acvt2(x), self.acvt2(n)
        return self.linear(torch.cat([x, n], dim=-1))


class Trans1d(nn.Module):
    '''
    Temporal Feature Transforming Layer for multi-channel 1D features.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='circular')

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        return self.transform(x, adj)

    def transform(self, x, adj):
        return self.conv(torch.einsum('bcxy,bncx->bncx', adj, x))

    def feature_adjacency(self, x, y):
        fadj = torch.einsum('bncx,bdcy->bcxy', x, y)
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(-1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class LFGN(FGN):
    def __init__(self, args,):
        super().__init__()
        self.args = args
        self.register_buffer('inputs', torch.Tensor(0, 5, 5, 50))
        self.register_buffer('targets', torch.LongTensor(0))
        self.sample_viewed = 0
        self.memory_order = torch.LongTensor()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.1)

    def observe(self, inputs, targets, reply=True):
        self.train()
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
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
                outputs = self.forward(inputs)
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
