
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
from torch.utils.tensorboard import SummaryWriter

from model import CNN
from ward import WARD
from torch_util import count_parameters, Timer, EarlyStopScheduler

sys.path.append('models')
warnings.filterwarnings("ignore")


def performance(loader, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(loader)):
            inputs, targets  = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
    return acc


def accuracy(net, loader, device, num_class):
    net.eval()
    correct, total = 0, 0
    classes = torch.arange(num_class).view(-1,1).to(device)
    with torch.no_grad():
        for idx, (inputs, targets, neighbor) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets, neighbor = inputs.to(device), targets.to(device), [item.to(device) for item in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += (targets == classes).sum(1)
            corrected = predicted==targets
            correct += torch.stack([corrected[targets==i].sum() for i in range(num_class)])
        acc = correct/total
    return acc


if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='accuracy/cora-lgl-test', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=50, help="epoch")
    parser.add_argument("--duration", type=int, default=50, help="duration")
    parser.add_argument("--batch-size", type=int, default=100, help="minibatch size")
    parser.add_argument("--jump", type=int, default=1, help="reply samples")
    parser.add_argument("--iteration", type=int, default=10, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=500, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--sample-rate", type=int, default=50, help="sampling rate for test acc, if ogb datasets please set it to 200")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    test_data = WARD(root=args.data_root, duration=args.duration, train=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data = WARD(root=args.data_root, duration=args.duration, train=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    writter = SummaryWriter()
    net, criterion = CNN().to(args.device), nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.1)
    scheduler = EarlyStopScheduler(optimizer, patience=3, factor=0.1, verbose=True, threshold=1e-1)
    print('Parameters: %d'%(count_parameters(net)))

    for epoch in range(args.epoch):
        for batch_idx, (inputs, target) in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            inputs, target = inputs.to(args.device), target.to(args.device)
            output = net(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            writter.add_scalar('loss', loss, epoch*len(train_loader)+batch_idx)
        acc = performance(test_loader, net, args.device)
        writter.add_scalar('accuracy', acc, epoch)
        print('Epoch: %d, Acc: %f'%(epoch, acc))
        if scheduler.step(1-acc):
            print('Early Stoping..')
            break
