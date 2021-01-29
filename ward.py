#!/usr/bin/env python3

import os
import wget
import tqdm
import torch
import os.path
import scipy.io
import numpy as np
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset


class WARD(VisionDataset):
    '''
    Wearable Action Recognition Dataset (WARD) for distributed pattern recognition.

    It is comprised of 20 human subjects (7 females and 13 males) with ages from 19 
    to 75 and 13 daily action categories, including rest at standing (ReSt), rest at
    sitting (ReSi), rest at lying (ReLi), walk forward (WaFo), walk forward left-circle
    (WaLe), walk forward right-circle (WaRi), turn left (TuLe), turn right (TuRi), go
    upstairs (Up), go downstairs (Down), jog (Jog), jump (Jump), and push wheelchair
    (Push). Five sensors, each of which consists of a triaxial accelerometer and a
    biaxial gyroscope, are located at the waist, left and right forearms, left and
    right ankles. Therefore, each sensor produces 5 dimensional data stream and
    totally 25 dimensional data is available. Each subject performs five trails
    for each activity, thus the database totally contains 20 × 13 × 5 data stream,
    each of which lasts more than 10 seconds and is recorded at 30Hz.

    The above description is from paper:
    [A] Chen Wang, Le Zhang, Lihua Xie, Junsong Yuan, Kernel Cross-Correlator (KCC), AAAI, 2018.

    The dataset is from:
    [B] Yang, Allen Y., et al. "Distributed recognition of human actions using wearable
    motion sensor networks." Journal of Ambient Intelligence and Smart Environments.
    '''
    url = 'https://github.com/wang-chen/KCC/releases/download/v1.0/ward.mat'
    def __init__(self, root='/data/datasets', duration=50, train=True):
        super().__init__(root)
        assert duration < 300
        self.duration = duration
        processed = os.path.join(root, 'WARD/ward.%s.torch'%('train' if train else 'test'))
        if not os.path.exists(processed):
            os.makedirs(os.path.join(root, 'WARD'), exist_ok=True)
            matfile = os.path.join(root, 'WARD/ward.mat')
            if not os.path.isfile(matfile):
                print('Downloading WARD dataset...')
                wget.download(self.url, matfile)
            sequence, size, data = [], [], scipy.io.loadmat(matfile)['data']
            for trial in range(6): # some subject has 6 trails
                for human in range(20):
                    for activity in range(13):
                        if data[human, activity, trial].size == 0:
                            continue
                        # Remove fisrt 75 unstable points as did in [A].
                        seq = torch.from_numpy(data[human, activity, trial])[75:,:].T.to(torch.float32)
                        seq = seq[:,:round(seq.size(-1)*0.8)] if train else seq[:,round(seq.size(-1)*0.8):]
                        sequence.append([seq.view(5,5,-1), activity])
                        size.append(torch.LongTensor([seq.size(-1)]))
            torch.save({'sequence':sequence, 'size':torch.cat(size)}, processed)
        data = torch.load(processed)
        self.sequence, self.size = data['sequence'], data['size']-duration
        self.cumsum = self.size.cumsum(dim=-1)

    def __len__(self):
        return sum(self.size)

    def __getitem__(self, ret):
        index = (self.cumsum - ret  >= 0).nonzero(as_tuple=False)[0]
        frame = self.cumsum[index] - ret
        sequence, target = self.sequence[index]
        return sequence[:,:,frame:frame+self.duration], target


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    data = WARD(root='/data/datasets', duration=50, train=True)
    loader = DataLoader(data, batch_size=5, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    for i, (sequence, target) in enumerate(loader):
        print(i, sequence.shape, target.T)