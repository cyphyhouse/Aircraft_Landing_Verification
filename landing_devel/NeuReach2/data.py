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
    def __init__(self, config, num_X0s=10000, num_traces=20, num_t=0, dim = 'x', use_data = True, data_file='data.pickle'):
        super(VisionData, self).__init__()

        self.perception = Perception()

        self.config = config 

        self.X0s = [self.config.sample_X0() for _ in range(num_X0s)]
        # self.X0_mean, self.X0_std = self.config.get_X0_normalization_factor()

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
        return torch.from_numpy(np.array(x).astype('float32')).view(-1),\
            torch.from_numpy(np.array(Ec).astype('float32')).view(-1),\
            torch.from_numpy(np.array(Er).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est_mean).astype('float32')).view(-1)

class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, config, num_X0s=100, num_traces=10, num_t=100, data_file=None):
        super(DiscriData, self).__init__()

        self.config = config

        self.X0s = [self.config.sample_X0() for _ in range(num_X0s)]
        self.X0_mean, self.X0_std = self.config.get_X0_normalization_factor()

        use_precomputed_data = os.path.exists(data_file) and use_data

        if use_precomputed_data:
            [self.traces, self.data] = loadpklz(data_file)
        else:
            func = partial(sample_trajs, num_traces, self.config.sample_x0, self.config.simulate, self.config.get_init_center)
            num_proc = min([1, multiprocessing.cpu_count()-3])
            # with Pool(num_proc, initializer=mute) as p:
                # self.traces = list(tqdm.tqdm(p.imap(func, self.X0s), total=len(self.X0s)))
            self.traces = list(tqdm.tqdm(map(func, self.X0s), total=len(self.X0s)))

            self.data = []
            for i in range(len(self.traces)):
                traces = self.traces[i]
                for j in range(traces.shape[0]-1):
                    sampled_ts = np.array([config.sample_t() for _ in range(num_t)]).reshape(-1,1)
                    ts = traces[j+1,:,0].reshape(1,-1)
                    idx_ts_j = np.abs(sampled_ts - ts).argmin(axis=1)
                    ts = traces[0,:,0].reshape(1,-1)
                    idx_ts_0 = np.abs(sampled_ts - ts).argmin(axis=1)

                    for (idx_t0, idx_tj, sampled_t) in zip(idx_ts_0, idx_ts_j, sampled_ts):
                        self.data.append([self.X0s[i], sampled_t, traces[0, idx_t0, 1:], traces[j+1, idx_tj, 1:]])
            if not use_precomputed_data:
                savepklz([self.traces, self.data], data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        X0 = data[0]
        t = data[1]
        ref = data[2]
        xt = data[3]
        return torch.from_numpy(((np.array(X0)-self.X0_mean)/self.X0_std).astype('float32')).view(-1),\
            torch.from_numpy(np.array(t).astype('float32')).view(-1),\
            torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(xt).astype('float32')).view(-1)

def get_dataloader(config, args):
    train_loader = torch.utils.data.DataLoader(
        DiscriData(config, args.N_X0, args.N_x0, args.N_t, data_file=args.data_file_train), batch_size=args.batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriData(config, args.num_test, data_file=args.data_file_eval), batch_size=args.batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    return train_loader, val_loader

def get_vision_dataloader(config, args):
    train_loader = torch.utils.data.DataLoader(
        VisionData(config, args.N_X0, args.N_x0, args.N_t, data_file=args.data_file_train), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        VisionData(config, args.num_test, data_file=args.data_file_eval), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    import autoland_system as AutoLand
    rospy.init_node('aircraft_landing')
    tmp = VisionData(config=AutoLand, data_file = 'data.pickle', use_data=False)