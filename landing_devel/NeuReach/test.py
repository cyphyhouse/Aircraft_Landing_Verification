import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

# script_dir = os.path.realpath(os.path.dirname(__file__))

# data_path = os.path.join(script_dir, 'data/data_backup.txt')
# label_path = os.path.join(script_dir, 'estimation_label/label_backup.txt')

data_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data_verif.txt'
label_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label_verif.txt'

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

model.load_state_dict(torch.load('./log/checkpoint.pth.tar')['state_dict'])

# data = torch.FloatTensor([-2936.190526247269, 23.028459769554445, 56.49611197902172, 0.041778978197086855, 0.0498730895584773, -0.013122412801362213])
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data = data[:,1:-1]
label = label[:,1:]

data_tensor = torch.FloatTensor(data)
label_tensor = torch.FloatTensor(label)

res = model.forward(data_tensor).detach().numpy()

plt.figure()
plt.plot(label[:,0],'b*')
plt.plot(data[:,0]+res[:,0],'r*')
plt.plot(data[:,0]-res[:,0],'r*')
plt.show()
# label = torch.FloatTensor([-2929.1353320511444, 20.64578387453148, 58.76066196314996, 0.04082988026075878, 0.05136111452277414, -0.012049659860212891])
# print(data)
# print(res)
# print(label)
    