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

import torch.utils.data as data
from utils import loadpklz, savepklz
from sklearn import preprocessing 


import copy

def mute():
    sys.stdout = open(os.devnull, 'w')

# def sample_trajs(num_traces, sample_x0, simulate, get_init_center, X0):
#     traces = []
#     traces.append(simulate(get_init_center(X0)))
#     for id_trace in range(num_traces):
#         traces.append(simulate(sample_x0(X0)))
#     return np.array(traces)

# class DiscriData(data.Dataset):
#     """DiscriData."""
#     def __init__(self, config, num_X0s=100, num_traces=10, num_t=100, data_file=None):
#         super(DiscriData, self).__init__()

#         self.config = config

#         self.X0s = [self.config.sample_X0() for _ in range(num_X0s)]
#         self.X0_mean, self.X0_std = self.config.get_X0_normalization_factor()

#         use_precomputed_data = os.path.exists(data_file)

#         if use_precomputed_data:
#             [self.traces, self.data] = loadpklz(data_file)
#         else:
#             func = partial(sample_trajs, num_traces, self.config.sample_x0, self.config.simulate, self.config.get_init_center)
#             num_proc = min([1, multiprocessing.cpu_count()-3])
#             # with Pool(num_proc, initializer=mute) as p:
#                 # self.traces = list(tqdm.tqdm(p.imap(func, self.X0s), total=len(self.X0s)))
#             self.traces = list(tqdm.tqdm(map(func, self.X0s), total=len(self.X0s)))

#             self.data = []
#             for i in range(len(self.traces)):
#                 traces = self.traces[i]
#                 for j in range(traces.shape[0]-1):
#                     sampled_ts = np.array([config.sample_t() for _ in range(num_t)]).reshape(-1,1)
#                     ts = traces[j+1,:,0].reshape(1,-1)
#                     idx_ts_j = np.abs(sampled_ts - ts).argmin(axis=1)
#                     ts = traces[0,:,0].reshape(1,-1)
#                     idx_ts_0 = np.abs(sampled_ts - ts).argmin(axis=1)

#                     for (idx_t0, idx_tj, sampled_t) in zip(idx_ts_0, idx_ts_j, sampled_ts):
#                         self.data.append([self.X0s[i], sampled_t, traces[0, idx_t0, 1:], traces[j+1, idx_tj, 1:]])
#             if not use_precomputed_data:
#                 savepklz([self.traces, self.data], data_file)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         data = self.data[index]
#         X0 = data[0]
#         t = data[1]
#         ref = data[2]
#         xt = data[3]
#         return torch.from_numpy(((np.array(X0)-self.X0_mean)/self.X0_std).astype('float32')).view(-1),\
#             torch.from_numpy(np.array(t).astype('float32')).view(-1),\
#             torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
#             torch.from_numpy(np.array(xt).astype('float32')).view(-1)

# def get_dataloader(config, args):
#     train_loader = torch.utils.data.DataLoader(
#         DiscriData(config, args.N_X0, args.N_x0, args.N_t, data_file=args.data_file_train), batch_size=args.batch_size, shuffle=True,
#         num_workers=20, pin_memory=True)

#     val_loader = torch.utils.data.DataLoader(
#         DiscriData(config, args.num_test, data_file=args.data_file_eval), batch_size=args.batch_size, shuffle=True,
#         num_workers=20, pin_memory=True)

#     return train_loader, val_loader



class DiscriDataAutoLand_Train(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=10, data_file=None):
        super(DiscriDataAutoLand_Train, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        self.data_train = data[:, 1:-1]
        self.label_train = label[:, 1:]
        # print(self.data_train)
        # print()


    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        index = int(np.floor(index/10))
        ref = self.data_train[self.num_sample*index, :]
        est = self.label_train[self.num_sample*index:self.num_sample*(index+1), :]
        return torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

class DiscriDataAutoLand_Train2(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Train2, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data3.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label3.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        self.data_train = data[:, 1:-1]
        self.label_train = label[:, 1:]
        # data_path = os.path.join(args.data_dir, 'data/data2_normalized.npy')
        # label_path = os.path.join(args.label_dir, 'estimation_label/label2_normalized.npy')
        # self.data_train = np.load(data_path)[:,:-1]
        # self.label_train = np.load(label_path)
        # print(self.data_train)
        # print()


    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        ref = self.data_train[index, :]
        est = self.label_train[index, :]
        return torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

class DiscriDataAutoLand_Verif(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=10, data_file=None):
        super(DiscriDataAutoLand_Verif, self).__init__()


        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data_verif.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label_verif.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        self.data_train = data[:, 1:-1]
        self.label_train = label[:, 1:]
        # print(self.data_train)

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        index = int(np.floor(index/10))
        ref = self.data_train[self.num_sample*index, :]
        est = self.label_train[self.num_sample*index:self.num_sample*(index+1), :]
        return torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

class DiscriDataAutoLand_Verif2(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Verif2, self).__init__()


        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data_verif.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label_verif.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        self.data_train = data[:, 1:-1]
        self.label_train = label[:, 1:]
        # print(self.data_train)

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        index = int(np.floor(index))
        ref = self.data_train[index, :]
        est = self.label_train[index, :]
        return torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

def get_dataloader_autoland(args):
    train_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Train(args, data_file=args.data_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Verif(args, data_file=args.label_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader

def get_dataloader_autoland2(args):
    train_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Train2(args, data_file=args.data_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Verif2(args, data_file=args.label_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader

class DiscriDataAutoLand_Train_dim(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Train_dim, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data4.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label4.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        data_length = data.shape[0]
        train_data_idx = int(data_length*0.9)
        if args.dimension == 'x':
            self.data_total = data[:train_data_idx, 1:2]
            self.ref_total = data[:train_data_idx, 1:2]
            self.label_total = label[:train_data_idx, 1:2]
        elif args.dimension == 'y':
            self.data_total = data[:train_data_idx, (1,2)]
            self.ref_total = data[:train_data_idx, 2:3]
            self.label_total = label[:train_data_idx, 2:3]
        elif args.dimension == 'z':
            self.data_total = data[:train_data_idx, (1,3)]
            self.ref_total = data[:train_data_idx, 3:4]
            self.label_total = label[:train_data_idx, 3:4]
        elif args.dimension == 'roll':
            self.data_total = data[:train_data_idx, (1,4)]
            self.ref_total = data[:train_data_idx, 4:5]
            self.label_total = label[:train_data_idx, 4:5]
        elif args.dimension == 'pitch':
            self.data_total = data[:train_data_idx, (1,5)]
            self.ref_total = data[:train_data_idx, 5:6]
            self.label_total = label[:train_data_idx, 5:6]
        elif args.dimension == 'yaw':
            self.data_total = data[:train_data_idx, (1,6)]
            self.ref_total = data[:train_data_idx, 6:7]
            self.label_total = label[:train_data_idx, 6:7]
        else:
            raise ValueError
        
        self.data_train = copy.deepcopy(self.data_total) 
        self.ref_train = copy.deepcopy(self.ref_total) 
        self.label_train = copy.deepcopy(self.label_total)

        self.fraction = args.fraction
        self.window_width = args.window_width
        self.dimension = args.dimension

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        data = self.data_train[index, :]
        ref = self.ref_train[index, :]
        est = self.label_train[index, :]
        return torch.from_numpy(np.array(data).astype('float32')).view(-1),\
            torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

    def reduce_data1(self):
        dist_array = np.abs(self.ref_total-self.label_total).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data2(self, forward_c):
        data_tensor = torch.FloatTensor(self.data_total).cuda()
        ref_tensor = torch.FloatTensor(self.ref_total).cuda()
        label_tensor = torch.FloatTensor(self.label_total).cuda()
        c = forward_c(data_tensor)
        dist_tensor = torch.abs(label_tensor - c)
        dist_array = dist_tensor.cpu().detach().numpy().squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data3(self):
        data_length = self.data_total.shape[0]
        data_max = np.max(self.data_total[:,0])
        data_min = np.min(self.data_total[:,0])
        bin_array = np.arange(data_min, data_max, self.window_width)
        data_dict = {}
        ref_dict = {}
        idx_dict = {}
        label_dict = {}
        for i in range(data_length):
            for bin_lb in bin_array:
                if self.data_total[i,0] > bin_lb and self.data_total[i,0] < bin_lb + self.window_width:
                    if bin_lb not in data_dict:
                        data_dict[bin_lb] = [self.data_total[i,0]]
                        ref_dict[bin_lb] = [self.ref_total[i,0]]
                        idx_dict[bin_lb] = [i]
                        label_dict[bin_lb] = [self.label_total[i,0]]
                    else:
                        data_dict[bin_lb].append(self.data_total[i,0])
                        ref_dict[bin_lb].append(self.ref_total[i,0])
                        idx_dict[bin_lb].append(i)
                        label_dict[bin_lb].append(self.label_total[i,0])
        kept_idx = []
        for key in data_dict:
            dist_array = np.abs(np.array(ref_dict[key])-np.array(label_dict[key]))
            sorted_dist_idx_array = np.argsort(dist_array)
            reduced_array = sorted_dist_idx_array[:round(sorted_dist_idx_array.size*self.fraction)]
            reduced_array = np.sort(reduced_array)
            kept_idx += (np.array(idx_dict[key])[reduced_array]).tolist()
        
        self.data_train = copy.deepcopy(self.data_total[kept_idx,:])
        self.ref_train = copy.deepcopy(self.ref_total[kept_idx,:])
        self.label_train = copy.deepcopy(self.label_total[kept_idx,:])

class DiscriDataAutoLand_Verif_dim(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Verif_dim, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data4.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label4.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        data_length = data.shape[0]
        train_data_idx = int(data_length*0.9)
        if args.dimension == 'x':
            self.data_total = data[train_data_idx:, 1:2]
            self.ref_total = data[train_data_idx:, 1:2]
            self.label_total = label[train_data_idx:, 1:2]
        elif args.dimension == 'y':
            self.data_total = data[train_data_idx:, (1,2)]
            self.ref_total = data[train_data_idx:, 2:3]
            self.label_total = label[train_data_idx:, 2:3]
        elif args.dimension == 'z':
            self.data_total = data[train_data_idx:, (1,3)]
            self.ref_total = data[train_data_idx:, 3:4]
            self.label_total = label[train_data_idx:, 3:4]
        elif args.dimension == 'roll':
            self.data_total = data[train_data_idx:, (1,4)]
            self.ref_total = data[train_data_idx:, 4:5]
            self.label_total = label[train_data_idx:, 4:5]
        elif args.dimension == 'pitch':
            self.data_total = data[train_data_idx:, (1,5)]
            self.ref_total = data[train_data_idx:, 5:6]
            self.label_total = label[train_data_idx:, 5:6]
        elif args.dimension == 'yaw':
            self.data_total = data[train_data_idx:, (1,6)]
            self.ref_total = data[train_data_idx:, 6:7]
            self.label_total = label[train_data_idx:, 6:7]
        else:
            raise ValueError

        self.data_train = copy.deepcopy(self.data_total) 
        self.ref_train = copy.deepcopy(self.ref_total) 
        self.label_train = copy.deepcopy(self.label_total)

        self.fraction = args.fraction
        self.window_width = args.window_width

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        data = self.data_train[index, :]
        ref = self.ref_train[index, :]
        est = self.label_train[index, :]
        return torch.from_numpy(np.array(data).astype('float32')).view(-1),\
            torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(est).astype('float32')).view(-1)

    def reduce_data1(self):
        dist_array = np.abs(self.ref_total-self.label_total).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data2(self, forward_c):
        data_tensor = torch.FloatTensor(self.data_total).cuda()
        ref_tensor = torch.FloatTensor(self.ref_total).cuda()
        label_tensor = torch.FloatTensor(self.label_total).cuda()
        c = forward_c(data_tensor)
        dist_tensor = torch.abs(label_tensor - c)
        dist_array = dist_tensor.cpu().detach().numpy().squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data3(self):
        data_length = self.data_total.shape[0]
        data_max = np.max(self.data_total[:,0])
        data_min = np.min(self.data_total[:,0])
        bin_array = np.arange(data_min, data_max, self.window_width)
        data_dict = {}
        ref_dict = {}
        idx_dict = {}
        label_dict = {}
        for i in range(data_length):
            for bin_lb in bin_array:
                if self.data_total[i,0] > bin_lb and self.data_total[i,0] < bin_lb + self.window_width:
                    if bin_lb not in data_dict:
                        data_dict[bin_lb] = [self.data_total[i,0]]
                        ref_dict[bin_lb] = [self.ref_total[i,0]]
                        idx_dict[bin_lb] = [i]
                        label_dict[bin_lb] = [self.label_total[i,0]]
                    else:
                        data_dict[bin_lb].append(self.data_total[i,0])
                        ref_dict[bin_lb].append(self.ref_total[i,0])
                        idx_dict[bin_lb].append(i)
                        label_dict[bin_lb].append(self.label_total[i,0])
        kept_idx = []
        for key in data_dict:
            dist_array = np.abs(np.array(ref_dict[key])-np.array(label_dict[key]))
            sorted_dist_idx_array = np.argsort(dist_array)
            reduced_array = sorted_dist_idx_array[:round(sorted_dist_idx_array.size*self.fraction)]
            reduced_array = np.sort(reduced_array)
            kept_idx += (np.array(idx_dict[key])[reduced_array]).tolist()
        
        self.data_train = copy.deepcopy(self.data_total[kept_idx,:])
        self.ref_train = copy.deepcopy(self.ref_total[kept_idx,:])
        self.label_train = copy.deepcopy(self.label_total[kept_idx,:])

def get_dataloader_autoland_dim(args):
    train_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Train_dim(args, data_file=args.data_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Verif_dim(args, data_file=args.label_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader

class DiscriDataAutoLand_Train_dim2(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Train_dim2, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data5.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label5.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        if args.dimension == 'x':
            self.data_total = data[0:22500, 1:2]
            self.ref_total = data[0:22500, 1:2]
            self.label_total = label[0:22500, 1:2]
        elif args.dimension == 'y':
            self.data_total = data[0:22500, (1,2)]
            self.ref_total = data[0:22500, 2:3]
            self.label_total = label[0:22500, 2:3]
        elif args.dimension == 'z':
            self.data_total = data[0:22500, (1,3)]
            self.ref_total = data[0:22500, 3:4]
            self.label_total = label[0:22500, 3:4]
        elif args.dimension == 'roll':
            self.data_total = data[0:22500, (1,4)]
            self.ref_total = data[0:22500, 4:5]
            self.label_total = label[0:22500, 4:5]
        elif args.dimension == 'pitch':
            self.data_total = data[0:22500, (1,5)]
            self.ref_total = data[0:22500, 5:6]
            self.label_total = label[0:22500, 5:6]
        elif args.dimension == 'yaw':
            self.data_total = data[0:22500, (1,6)]
            self.ref_total = data[0:22500, 6:7]
            self.label_total = label[0:22500, 6:7]
        else:
            raise ValueError
        
        self.data_train = copy.deepcopy(self.data_total) 
        self.ref_train = copy.deepcopy(self.ref_total) 
        self.label_train = copy.deepcopy(self.label_total)

        self.fraction = args.fraction

    def __len__(self):
        return int(len(self.data_train)/50)

    def __getitem__(self, index):
        data = self.data_train[index*50:(index+1)*50, :]
        ref = self.ref_train[index*50:(index+1)*50, :]
        est = self.label_train[index*50:(index+1)*50, :]
        ref_mean = np.mean(ref, axis=0)
        dist_array = np.abs(ref - ref_mean).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        data = copy.deepcopy(data[reduced_array,:])
        ref = copy.deepcopy(ref[reduced_array,:])
        est = copy.deepcopy(est[reduced_array,:])

        if data.size==0:
            print("stop here")
        
        return torch.from_numpy(np.array(data).astype('float32')),\
            torch.from_numpy(np.array(ref).astype('float32')),\
            torch.from_numpy(np.array(est).astype('float32'))

    def reduce_data1(self):
        dist_array = np.abs(self.ref_total-self.label_total).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data2(self, forward_c):
        data_tensor = torch.FloatTensor(self.data_total).cuda()
        ref_tensor = torch.FloatTensor(self.ref_total).cuda()
        label_tensor = torch.FloatTensor(self.label_total).cuda()
        c = forward_c(data_tensor)
        dist_tensor = torch.abs(label_tensor - c)
        dist_array = dist_tensor.cpu().detach().numpy().squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

class DiscriDataAutoLand_Verif_dim2(data.Dataset):
    """DiscriData."""
    def __init__(self, args, num_sample=1, data_file=None):
        super(DiscriDataAutoLand_Verif_dim2, self).__init__()

        self.num_sample = num_sample

        data_path = os.path.join(args.data_dir, 'data/data4.txt')
        label_path = os.path.join(args.label_dir, 'estimation_label/label4.txt')
        data = np.loadtxt(data_path, delimiter=',')
        label = np.loadtxt(label_path, delimiter=',')
        if args.dimension == 'x':
            self.data_total = data[22500:25000, 1:2]
            self.ref_total = data[22500:25000, 1:2]
            self.label_total = label[22500:25000, 1:2]
        elif args.dimension == 'y':
            self.data_total = data[22500:25000, (1,2)]
            self.ref_total = data[22500:25000, 2:3]
            self.label_total = label[22500:25000, 2:3]
        elif args.dimension == 'z':
            self.data_total = data[22500:25000, (1,3)]
            self.ref_total = data[22500:25000, 3:4]
            self.label_total = label[22500:25000, 3:4]
        elif args.dimension == 'roll':
            self.data_total = data[22500:25000, (1,4)]
            self.ref_total = data[22500:25000, 4:5]
            self.label_total = label[22500:25000, 4:5]
        elif args.dimension == 'pitch':
            self.data_total = data[22500:25000, (1,5)]
            self.ref_total = data[22500:25000, 5:6]
            self.label_total = label[22500:25000, 5:6]
        elif args.dimension == 'yaw':
            self.data_total = data[22500:25000, (1,6)]
            self.ref_total = data[22500:25000, 6:7]
            self.label_total = label[22500:25000, 6:7]
        else:
            raise ValueError

        self.data_train = copy.deepcopy(self.data_total) 
        self.ref_train = copy.deepcopy(self.ref_total) 
        self.label_train = copy.deepcopy(self.label_total)

        self.fraction = args.fraction

    def __len__(self):
        return int(len(self.data_train)/50)

    def __getitem__(self, index):
        data = self.data_train[index*50:(index+1)*50, :]
        ref = self.ref_train[index*50:(index+1)*50, :]
        est = self.label_train[index*50:(index+1)*50, :]
        ref_mean = np.mean(ref, axis=0)
        dist_array = np.abs(ref - ref_mean).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        data = copy.deepcopy(data[reduced_array,:])
        ref = copy.deepcopy(ref[reduced_array,:])
        est = copy.deepcopy(est[reduced_array,:])
        
        return torch.from_numpy(np.array(data).astype('float32')),\
            torch.from_numpy(np.array(ref).astype('float32')),\
            torch.from_numpy(np.array(est).astype('float32'))

    def reduce_data1(self):
        dist_array = np.abs(self.ref_total-self.label_total).squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

    def reduce_data2(self, forward_c):
        data_tensor = torch.FloatTensor(self.data_total).cuda()
        ref_tensor = torch.FloatTensor(self.ref_total).cuda()
        label_tensor = torch.FloatTensor(self.label_total).cuda()
        c = forward_c(data_tensor)
        dist_tensor = torch.abs(label_tensor - c)
        dist_array = dist_tensor.cpu().detach().numpy().squeeze()
        sorted_dist_idx_array = np.argsort(dist_array)
        reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*self.fraction)]
        reduced_array = np.sort(reduced_array)
        
        self.data_train = copy.deepcopy(self.data_total[reduced_array,:])
        self.ref_train = copy.deepcopy(self.ref_total[reduced_array,:])
        self.label_train = copy.deepcopy(self.label_total[reduced_array,:])

def get_dataloader_autoland_dim2(args):
    train_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Train_dim2(args, data_file=args.data_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriDataAutoLand_Verif_dim2(args, data_file=args.label_dir), batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader
