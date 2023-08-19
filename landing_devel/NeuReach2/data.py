from __future__ import print_function
import multiprocessing
from multiprocessing import Pool
from functools import partial
import tqdm
import os
import os.path
import errno
import numpy as np
import sys
import torch
from autoland_system import Perception

import torch.utils.data as data
from utils.utils import loadpklz, savepklz
import rospy

import pickle

def mute():
    sys.stdout = open(os.devnull, 'w')

def sample_trajs(num_traces, sample_x0, simulate, get_init_center, perception, X0):
    traces = []
    traces.append(simulate(get_init_center(X0), perception))
    for id_trace in range(num_traces):
        traces.append(simulate(sample_x0(X0), perception))
    return np.array(traces)

class VisionData(data.Dataset):
    def __init__(self, config, num_X0s=10000, num_traces=20, num_t=0, dim = None, use_data = True, data_file='data.pickle'):
        super(VisionData, self).__init__()

        self.dim = dim

        self.perception = Perception()

        self.config = config 

        # self.X0s = [self.config.sample_X0() for _ in range(num_X0s)]
        self.X0s = []
        for _ in range(num_X0s):
            X01, X02 = self.config.sample_2X0()
            self.X0s.append(X01)
            self.X0s.append(X02)

        use_precomputed_data = False
        if data_file is not None:
            use_precomputed_data = os.path.exists(data_file) and use_data

        if use_precomputed_data:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            func = partial(sample_trajs, num_traces, self.config.sample_x0, self.config.simulate, self.config.get_init_center, self.perception)
            traces = list(tqdm.tqdm(map(func, self.X0s), total=len(self.X0s)))
            self.data = []
            for i in range(len(self.X0s)):
                trace = traces[i]
                self.data.append((self.X0s[i], trace.reshape((-1,1))))
            if data_file is not None:
                with open(data_file, 'wb+') as f:
                    pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        X0 = data[0]
        x, Ec, Er = X0
        est = data[1]
        est_mean = np.reshape(est, (-1,6))
        est_mean = np.mean(est_mean, axis=0)
        if self.dim is not None:
            if self.dim == 'x':
                x_dim = [x[0]]
                est_dim = np.reshape(est, (-1,6))
                est_dim = est_dim[:, 0:1]
                est_dim = np.reshape(est_dim, (-1,1))
                est_mean_dim = est_mean[0:1]
            elif self.dim=='y':
                x_dim = [np.array(x)[(0,1),]]
                est_dim = np.reshape(est, (-1,6))
                est_dim = est_dim[:, (0,1)]
                est_dim = np.reshape(est_dim, (-1,1))
                est_mean_dim = est_mean[(0,1),]
        return torch.from_numpy(np.array(x_dim).astype('float32')).view(-1),\
            torch.from_numpy(np.array(Ec).astype('float32')).view(-1),\
            torch.from_numpy(np.array(Er).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est_dim).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est_mean_dim).astype('float32')).view(-1)

def get_vision_dataloader(config, args):
    train_loader = torch.utils.data.DataLoader(
        VisionData(config, args.N_X0, args.N_x0, args.N_t, data_file=args.data_file_train, dim=args.dim), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)

    # val_loader = None
    val_loader = torch.utils.data.DataLoader(
        VisionData(config, args.num_test, data_file=args.data_file_eval, dim=args.dim), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    import autoland_system as AutoLand
    rospy.init_node('aircraft_landing')
    tmp = VisionData(num_X0s = 10000, num_traces = 20, config=AutoLand, data_file = 'data_train.pickle', use_data=False)