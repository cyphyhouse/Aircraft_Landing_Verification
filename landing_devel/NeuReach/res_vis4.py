import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from model import get_model_rect2, get_model_rect, get_model_rect3
from sklearn import preprocessing
import sys 

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, '../data/data4.txt')
label_path = os.path.join(script_dir, '../estimation_label/label4.txt')

# def get_model_rect2(num_dim_input, num_dim_output, layer1, layer2):
#     global mult
#     model = torch.nn.Sequential(
#             torch.nn.Linear(num_dim_input, layer1, bias=False),
#             torch.nn.Tanh(),
#             torch.nn.Linear(layer1, layer2, bias=False),
#             torch.nn.Tanh(),
#             torch.nn.Linear(layer2, num_dim_output, bias=False))

#     mult = None

#     def forward(input):
#         global mult
#         output = model(input)
#         output = output.view(input.shape[0], num_dim_output)
#         if mult is not None:
#             mult = mult.type(input.type())
#             output = torch.matmul(output, mult)
#         return output
#     return model, forward

# model_r_name = 'checkpoint_x_r_06-26_10-46-59_39.pth.tar'
# model_c_name = 'checkpoint_x_c_06-26_10-46-59_39.pth.tar'
model_r_name = 'checkpoint_z_r_06-28_09-34-27_65.pth.tar'
model_c_name = 'checkpoint_z_c_06-28_09-34-27_65.pth.tar'
tmp = model_r_name.split('_')
dim = tmp[1]

if dim == 'x':
    model_r, forward_r = get_model_rect2(1,1,64,64,64)
    model_c, forward_c = get_model_rect(1,1,64,64)
else:
    model_r, forward_r = get_model_rect2(2,1,64,64,64)
    model_c, forward_c = get_model_rect(2,1,64,64)
# model, forward = get_model_rect(6,6,32,32)

model_r.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_c.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_c_name}'), map_location=torch.device('cpu'))['state_dict'])

# data = torch.FloatTensor([-2936.190526247269, 23.028459769554445, 56.49611197902172, 0.041778978197086855, 0.0498730895584773, -0.013122412801362213])
data_orig = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data_orig = data_orig[90000:, :]
label = label[90000:, :]

if dim == 'x':
    data = data_orig[:,1:2]
    label = label[:,1:2]
elif dim == 'y':
    data = data_orig[:,[1,2]]
    label = label[:, 2:3]
elif dim == 'z':
    data = data_orig[:,[1,3]]
    label = label[:, 3:4]
elif dim == 'roll':
    data = data_orig[:,[1,4]]
    label = label[:,4:5]
elif dim == 'pitch':
    data = data_orig[:,[1,5]]
    label = label[:,5:6]
elif dim == 'yaw':
    data = data_orig[:,[1,6]]
    label = label[:,6:7]

data_tensor = torch.FloatTensor(data)
label_tensor = torch.FloatTensor(label)

res_r = model_r.forward(data_tensor).detach().numpy()
res_r = np.abs(res_r)
res_c = model_c.forward(data_tensor).detach().numpy()
# res = scaler_label_train.inverse_transform(res)

if dim == 'x':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth x")
    plt.ylabel('estimated x')
    # plt.savefig('surrogate_bound_x.png')
elif dim == 'y':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth x")
    plt.ylabel('estimated y')
    plt.savefig('surrogate_bound_y.png')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data[:,0], data[:,1], label[:,0], c='b', marker='*')
    # ax.scatter(data[:,0], data[:,1], res_c+res_r, c='r', marker='*')
    # ax.scatter(data[:,0], data[:,1], res_c-res_r, c='r', marker='*')
    # ax.set_xlabel('ground truth x')
    # ax.set_ylabel('ground truth y')
    # ax.set_zlabel('estimate y')
elif dim == 'z':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth x")
    plt.ylabel('estimated z')
    # plt.savefig('surrogate_bound_z.png')
elif dim == 'roll':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth roll")
    plt.ylabel('estimated roll')
    # plt.savefig('surrogate_bound_roll.png')
elif dim == 'pitch':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth pitch")
    plt.ylabel('estimated pitch')
    # plt.savefig('surrogate_bound_pitch.png')
elif dim == 'yaw':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], res_c+res_r,'r*')
    plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth yaw")
    plt.ylabel('estimated yaw')
    # plt.savefig('surrogate_bound_yaw.png')

plt.show()
# label = torch.FloatTensor([-2929.1353320511444, 20.64578387453148, 58.76066196314996, 0.04082988026075878, 0.05136111452277414, -0.012049659860212891])
# print(data)
# print(res)
# print(label)
    