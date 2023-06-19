import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from model import get_model_rect2
from sklearn import preprocessing


script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, '../data/data_verif.txt')
label_path = os.path.join(script_dir, '../estimation_label/label_verif.txt')

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

model, forward = get_model_rect2(6, 6, 128, 128, 128)

model.load_state_dict(torch.load(os.path.join(script_dir, './log/checkpoint_1_06-19.pth.tar'), map_location=torch.device('cpu'))['state_dict'])

# data = torch.FloatTensor([-2936.190526247269, 23.028459769554445, 56.49611197902172, 0.041778978197086855, 0.0498730895584773, -0.013122412801362213])
data_orig = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data = data_orig[:,1:-1]
label = label[:,1:]

data_train_path = "/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data2.txt"
label_train_path = "/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label2.txt"

data_train = np.loadtxt(data_train_path, delimiter=',')
label_train = np.loadtxt(label_train_path, delimiter=',')
data_train = data_train[:, 1:-1]
label_train = label_train[:, 1:]

# print(data_train)

scaler_data_train = preprocessing.StandardScaler().fit(data_train)
# data_train_normalized = scaler_data.transform(data_train)
# # print(data_train_normalized)
# scaler_label_train = preprocessing.StandardScaler().fit(label_train)
# label_train_normalized = scaler_label.transform(label_train)

data = scaler_data_train.transform(data)
label = scaler_data_train.transform(label)

data_tensor = torch.FloatTensor(data)
label_tensor = torch.FloatTensor(label)

res = model.forward(data_tensor).detach().numpy()
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

plt.figure()
plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
plt.plot(data[:,0], data[:,0]+res[:,0],'r*')
plt.plot(data[:,0], data[:,0]-res[:,0],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth x")
plt.ylabel('estimated x')
plt.savefig('surrogate_bound_x.png')

plt.figure()
plt.plot(data[:,1], label[:,1],'b*', label='estimated state')
plt.plot(data[:,1], data[:,1]+res[:,1],'r*')
plt.plot(data[:,1], data[:,1]-res[:,1],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth y")
plt.ylabel('estimated y')
plt.savefig('surrogate_bound_y.png')

plt.figure()
plt.plot(data[:,2], label[:,2],'b*', label='estimated state')
plt.plot(data[:,2], data[:,2]+res[:,2],'r*')
plt.plot(data[:,2], data[:,2]-res[:,2],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth z")
plt.ylabel('estimated z')
plt.savefig('surrogate_bound_z.png')

plt.figure()
plt.plot(data[:,3], label[:,3],'b*', label='estimated state')
plt.plot(data[:,3], data[:,3]+res[:,3],'r*')
plt.plot(data[:,3], data[:,3]-res[:,3],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth roll")
plt.ylabel('estimated roll')
plt.savefig('surrogate_bound_roll.png')

plt.figure()
plt.plot(data[:,4], label[:,4],'b*', label='estimated state')
plt.plot(data[:,4], data[:,4]+res[:,4],'r*')
plt.plot(data[:,4], data[:,4]-res[:,4],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth pitch")
plt.ylabel('estimated pitch')
plt.savefig('surrogate_bound_pitch.png')

plt.figure()
plt.plot(data[:,5], label[:,5],'b*', label='estimated state')
plt.plot(data[:,5], data[:,5]+res[:,5],'r*')
plt.plot(data[:,5], data[:,5]-res[:,5],'r*', label='surrogate bound')
plt.legend()
plt.xlabel("ground truth yaw")
plt.ylabel('estimated yaw')
plt.savefig('surrogate_bound_yaw.png')

plt.show()
# label = torch.FloatTensor([-2929.1353320511444, 20.64578387453148, 58.76066196314996, 0.04082988026075878, 0.05136111452277414, -0.012049659860212891])
# print(data)
# print(res)
# print(label)
    