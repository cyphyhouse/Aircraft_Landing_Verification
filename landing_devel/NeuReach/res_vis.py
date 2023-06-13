import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, '../data/data_verif.txt')
label_path = os.path.join(script_dir, '../estimation_label/label_verif.txt')

def get_model_rect(num_dim_input, num_dim_output, layer1, layer2):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, layer1, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(layer1, layer2, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(layer2, num_dim_output, bias=False))

    mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward

model, forward = get_model_rect(6, 6, 64, 64)

model.load_state_dict(torch.load(os.path.join(script_dir, './log/checkpoint_1.pth.tar'))['state_dict'])

# data = torch.FloatTensor([-2936.190526247269, 23.028459769554445, 56.49611197902172, 0.041778978197086855, 0.0498730895584773, -0.013122412801362213])
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data = data[:,1:-1]
label = label[:,1:]

data_tensor = torch.FloatTensor(data)
label_tensor = torch.FloatTensor(label)

res = model.forward(data_tensor).detach().numpy()

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
    