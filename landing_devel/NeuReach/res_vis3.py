import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from model import get_model_rect2, get_model_rect
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

model_name = 'checkpoint_y_06-23_15-03-03_2.pth.tar'
tmp = model_name.split('_')
dim = tmp[1]

if dim == 'x':
    model, forward = get_model_rect2(1, 2, 256, 256, 256)
else:
    model, forward = get_model_rect(2,2,64,64)
# model, forward = get_model_rect(6,6,32,32)

model.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_name}'), map_location=torch.device('cpu'))['state_dict'])

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

res = model.forward(data_tensor).detach().numpy()
res = np.abs(res)
# res = scaler_label_train.inverse_transform(res)

total_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

mis_classify = 0
mis_classify_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

for i in range(data.shape[0]):
    if data_orig[i,-1] >=0.5 and data_orig[i,-1]<0.6:
        total_dict[0] += 1
    elif data_orig[i,-1] >=0.6 and data_orig[i,-1]<0.7:
        total_dict[1] += 1
    elif data_orig[i,-1] >=0.7 and data_orig[i,-1]<0.8:
        total_dict[2] += 1
    elif data_orig[i,-1] >=0.8 and data_orig[i,-1]<0.9:
        total_dict[3] += 1
    elif data_orig[i,-1] >=0.9 and data_orig[i,-1]<1.0:
        total_dict[4] += 1
    elif data_orig[i,-1] >=1.0 and data_orig[i,-1]<1.1:
        total_dict[5] += 1
    else:
        total_dict[6] += 1
    for j in range(data.shape[1]):
        lb, ub = data[i,j]-abs(res[i,j]), data[i,j]+abs(res[i,j])
        if label[i,j]<lb or label[i,j]>ub:
            mis_classify += 1 
            if data_orig[i,-1] >=0.5 and data_orig[i,-1]<0.6:
                mis_classify_dict[0] += 1
            elif data_orig[i,-1] >=0.6 and data_orig[i,-1]<0.7:
                mis_classify_dict[1] += 1
            elif data_orig[i,-1] >=0.7 and data_orig[i,-1]<0.8:
                mis_classify_dict[2] += 1
            elif data_orig[i,-1] >=0.8 and data_orig[i,-1]<0.9:
                mis_classify_dict[3] += 1
            elif data_orig[i,-1] >=0.9 and data_orig[i,-1]<1.0:
                mis_classify_dict[4] += 1
            elif data_orig[i,-1] >=1.0 and data_orig[i,-1]<1.1:
                mis_classify_dict[5] += 1
            else:
                mis_classify_dict[6] += 1
            break

print(data.shape[0])
print(total_dict)
print(mis_classify)
print(mis_classify_dict)

if dim == 'x':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,0]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,0]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth x")
    plt.ylabel('estimated x')
    # plt.savefig('surrogate_bound_x.png')
elif dim == 'y':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,1]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,1]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth y")
    plt.ylabel('estimated y')
    plt.savefig('surrogate_bound_y.png')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data[:,0], data[:,1], label[:,0], c='b', marker='*')
    # ax.scatter(data[:,0], data[:,1], data[:,1]+res[:,0], c='r', marker='*')
    # ax.scatter(data[:,0], data[:,1], data[:,1]-res[:,1], c='r', marker='*')
    # ax.set_xlabel('ground truth x')
    # ax.set_ylabel('ground truth y')
    # ax.set_zlabel('estimate y')
elif dim == 'z':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,1]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,1]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth z")
    plt.ylabel('estimated z')
    # plt.savefig('surrogate_bound_z.png')
elif dim == 'roll':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,1]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,1]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth roll")
    plt.ylabel('estimated roll')
    # plt.savefig('surrogate_bound_roll.png')
elif dim == 'pitch':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,1]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,1]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth pitch")
    plt.ylabel('estimated pitch')
    # plt.savefig('surrogate_bound_pitch.png')
elif dim == 'yaw':
    plt.figure()
    plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
    plt.plot(data[:,0], data[:,1]+res[:,0],'r*')
    plt.plot(data[:,0], data[:,1]-res[:,1],'r*', label='surrogate bound')
    plt.legend()
    plt.xlabel("ground truth yaw")
    plt.ylabel('estimated yaw')
    # plt.savefig('surrogate_bound_yaw.png')

plt.show()
# label = torch.FloatTensor([-2929.1353320511444, 20.64578387453148, 58.76066196314996, 0.04082988026075878, 0.05136111452277414, -0.012049659860212891])
# print(data)
# print(res)
# print(label)
    