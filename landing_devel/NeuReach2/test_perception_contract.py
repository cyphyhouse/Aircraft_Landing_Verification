import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils.utils import AverageMeter

from data import get_vision_dataloader
from model import get_model_rect
from datetime import datetime 
import pickle 

import matplotlib.pyplot as plt 

start_time = datetime.now()
start_time_str = start_time.strftime("%m-%d_%H-%M-%S")

import sys
sys.path.append('systems')

import argparse

dim = 'x'

script_dir = os.path.dirname(os.path.realpath(__file__))

model_r_x_name = os.path.join(script_dir, './log/checkpoint_x_r_08-17_14-18-49_775.pth.tar')
model_c_x_name = os.path.join(script_dir, './log/checkpoint_x_c_08-17_14-18-49_775.pth.tar')

train_data_fn = os.path.join(script_dir, 'data.pickle')
test_data_fn = os.path.join(script_dir, 'data_eval.pickle')

import autoland_system as config
x, Ec, Er = config.sample_X0()
X0 = np.concatenate((x, Ec, Er))
model_x_r, forward_x_r = get_model_rect(3, 1, 64, 64)
model_x_c, forward_x_c = get_model_rect(3, 1, 64, 64)
model_x_r.load_state_dict(torch.load(model_r_x_name, map_location=torch.device('cpu'))['state_dict'])
model_x_c.load_state_dict(torch.load(model_c_x_name, map_location=torch.device('cpu'))['state_dict'])

with open(train_data_fn, 'rb') as f:
    train_data = pickle.load(f) 

with open(test_data_fn, 'rb') as f:
    test_data = pickle.load(f)

state_list = []
Er_list = []
Ec_list = []
est = []
for data_point in train_data:
    X0, Est = data_point
    state, Ec, Er = X0 
    state_list.append(state)
    Er_list.append(Er)
    Ec_list.append(Ec)
    est.append(Est)

state_list = np.array(state_list)
Er_list = np.array(Er_list)
Ec_list = np.array(Ec_list)
x_list = state_list[:,0:1]

x_tensor = torch.FloatTensor(x_list)
Er_tensor = torch.FloatTensor(Er_list)
Ec_tensor = torch.FloatTensor(Ec_list)

input_tensor = torch.cat([x_tensor, Ec_tensor, Er_tensor], axis=1)
print(input_tensor.shape)
output_x_r = model_x_r(input_tensor)
output_x_c = model_x_c(input_tensor)

output_x_r = output_x_r.detach().numpy()
output_x_c = output_x_c.detach().numpy()

bin = []
bin_lb_list, bin_step = np.linspace(0, 0.35, 51, retstep=True)
r_list = output_x_r
Er_center_list = []
bound_list = []
for i in range(len(bin_lb_list)-1):
    bin_lb = bin_lb_list[i]
    bin_ub = bin_lb + bin_step 
    idx = np.where((bin_lb<Er_list) & (Er_list<bin_ub))[0]
    Er_center_list.append((bin_lb+bin_ub)/2)
    r_sub_list = r_list[idx]
    percentile = np.percentile(r_sub_list, 80)
    bound_list.append(percentile)

plt.plot(Er_center_list, bound_list, '*')
plt.show()

