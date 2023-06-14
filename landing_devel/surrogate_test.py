import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import os 

class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(SurrogateModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 32)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 32)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(32, 6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        return x

script_dir = os.path.realpath(os.path.dirname(__file__))

ground_truth_position = np.load(os.path.join(script_dir, 'ground_truth.npy'), allow_pickle=True)
estimate_pos = np.load(os.path.join(script_dir, 'estimation.npy'), allow_pickle=True)

ground_truth_state = []
for i in range(len(ground_truth_position[:, 0])):
    # print(ground_truth_position[i])
    ground_truth_state.append(np.array([ground_truth_position[i][0][0], ground_truth_position[i][1][0], ground_truth_position[i][2][0], ground_truth_position[i][3], ground_truth_position[i][4][0], ground_truth_position[i][5][0]])) 
    # print(state)

ground_truth_state = np.array(ground_truth_state)


data_path = os.path.join(script_dir, 'data/data_verif.txt')
label_path = os.path.join(script_dir, 'estimation_label/label_verif.txt')
x = np.loadtxt(data_path, delimiter=',')
y = np.loadtxt(label_path, delimiter=',')

ground_truth_state=x[:, 1:]
estimate_pos = y[:, 1:]

ground_truth_state_tensor = torch.FloatTensor(ground_truth_state)
estimate_pos_tensor = torch.FloatTensor(estimate_pos)

input_dim = ground_truth_state_tensor.shape[1]
surrogate_model = SurrogateModel(input_dim)

surrogate_model.load_state_dict(torch.load(os.path.join(script_dir, 'surrogate_model_06-05_lighting.pth')))
surrogate_model.eval()
predict_pos = surrogate_model(ground_truth_state_tensor)
predict_pos = predict_pos.detach().numpy()

font = {'family': 'serif', 'weight': 'normal', 'size': 14}

plt.figure()
# plt.plot(ground_truth_state[:,0],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,0], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,0], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,0], predict_pos[:,0], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,0], estimate_pos[:,0], 'b*', label='Estimation')
plt.xlabel('Ground truth x', fontdict=font)
plt.ylabel('Estimated x')
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_x.png'))

plt.figure()
plt.plot(ground_truth_state[:,0], abs(predict_pos[:,0]-estimate_pos[:,0]), 'g*')
plt.xlabel("Ground truth x", fontdict=font)
plt.ylabel("x error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_x.png'))

plt.figure()
# plt.plot(ground_truth_state[:,1],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,1], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,1], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,1], predict_pos[:,1], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,1], estimate_pos[:,1], 'b*', label='Estimation')
plt.xlabel('Ground truth y', fontdict=font)
plt.ylabel('Estimated y', fontdict=font)
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_y.png'))

plt.figure()
plt.plot(ground_truth_state[:,1], abs(predict_pos[:,1]-estimate_pos[:,1]), 'g*')
plt.xlabel("Ground truth y", fontdict=font)
plt.ylabel("y error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_y.png'))

plt.figure()
# plt.plot(ground_truth_state[:,2],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,2], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,2], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,2], predict_pos[:,2], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,2], estimate_pos[:,2], 'b*', label='Estimation')
plt.xlabel('Ground truth z', fontdict=font)
plt.ylabel('Estimated z', fontdict=font)
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_z.png'))

plt.figure()
plt.plot(ground_truth_state[:,2], abs(predict_pos[:,2]-estimate_pos[:,2]), 'g*')
plt.xlabel("Ground truth", fontdict=font)
plt.ylabel("z error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_z.png'))

plt.figure()
# plt.plot(ground_truth_state[:,3],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,3], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,3], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,3], predict_pos[:,3], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,3], estimate_pos[:,3], 'b*', label='Estimation')
plt.xlabel('Ground truth roll', fontdict=font)
plt.ylabel('Estimated roll', fontdict=font)
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_roll.png'))

plt.figure()
plt.plot(ground_truth_state[:,3], abs(predict_pos[:,3]-estimate_pos[:,3]), 'g*')
plt.xlabel("Ground truth roll", fontdict=font)
plt.ylabel("Roll error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_roll.png'))

plt.figure()
# plt.plot(ground_truth_state[:,4],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,4], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,4], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,4], predict_pos[:,4], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,4], estimate_pos[:,4], 'b*', label='Estimation')
plt.xlabel('Ground truth pitch', fontdict=font)
plt.ylabel('Estimated pitch', fontdict=font)
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_pitch.png'))

plt.figure()
plt.plot(ground_truth_state[:,4], abs(predict_pos[:,4]-estimate_pos[:,4]), 'g*')
plt.xlabel("Ground truth pitch", fontdict=font)
plt.ylabel("Pitch error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_pitch.png'))

plt.figure()
# plt.plot(ground_truth_state[:,5],'g*', label='Ground Truth')
# plt.plot(predict_pos[:,5], 'r*', label='Surrogate Estimation')
# plt.plot(estimate_pos[:,5], 'b*', label='Estimation')
plt.plot(ground_truth_state[:,5], predict_pos[:,5], 'r*', label='Surrogate Estimation')
plt.plot(ground_truth_state[:,5], estimate_pos[:,5], 'b*', label='Estimation')
plt.xlabel('Ground truth yaw', fontdict=font)
plt.ylabel('Estimated yaw', fontdict=font)
plt.legend()
plt.savefig(os.path.join(script_dir,'surrogate_compare_yaw.png'))

plt.figure()
plt.plot(ground_truth_state[:,5], abs(predict_pos[:,5]-estimate_pos[:,5]), 'g*')
plt.xlabel("Ground truth yaw", fontdict=font)
plt.ylabel("Yaw error", fontdict=font)
plt.savefig(os.path.join(script_dir,'surrogate_error_yaw.png'))

plt.show()