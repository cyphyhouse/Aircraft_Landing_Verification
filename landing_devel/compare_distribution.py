import numpy as np 
import matplotlib.pyplot as plt 
import os 

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, 'data/data.txt')
label_path = os.path.join(script_dir, 'estimation_label/label.txt')
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')
# table = [row.strip().split('\n') for row in x]

data_train = data[:, 1:]
label_train = label[:, 1:]
# tmp = data[8000:, 1:]
# label_test = label[8000:, 1:]

tmp = np.load(os.path.join(script_dir, 'ground_truth.npy'), allow_pickle=True)
label_test = np.load(os.path.join(script_dir, 'estimation.npy'), allow_pickle=True)

# tmp = np.array(tmp)
# print(len(tmp))
# estimate_pos = np.array(estimate_pos).squeeze()

data_test = []

for i in range(len(tmp[:, 0])):
    # print(tmp[i])
    data_test.append(np.array([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]])) 
    # print(state)
data_test = np.array(data_test)

# idx_list = np.where(data_train[:,0]<-3150)
# print(idx_list)
# plt.figure()
# plt.plot(data_train[idx_list,0],label_train[idx_list,0], 'b*')
# plt.show()

plt.figure()
plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
plt.plot(data_test[:,0], label_test[:,0],'r*', label='Testing Set')
plt.xlabel('Ground Truth x')
plt.ylabel('Estimate x')
plt.legend()
plt.savefig('distribution_x.png')

plt.figure()
plt.plot(data_train[:,1], label_train[:,1],'b*', label='Training Set')
plt.plot(data_test[:,1], label_test[:,1],'r*', label='Testing Set')
plt.xlabel('Ground Truth y')
plt.ylabel('Estimate y')
plt.legend()
plt.savefig('distribution_y.png')

plt.figure()
plt.plot(data_train[:,2], label_train[:,2],'b*', label='Training Set')
plt.plot(data_test[:,2], label_test[:,2],'r*', label='Testing Set')
plt.xlabel('Ground Truth z')
plt.ylabel('Estimate z')
plt.legend()
plt.savefig('distribution_z.png')

plt.figure()
plt.plot(data_train[:,3], label_train[:,3],'b*', label='Training Set')
plt.plot(data_test[:,3], label_test[:,3],'r*', label='Testing Set')
plt.xlabel('Ground Truth roll')
plt.ylabel('Estimate roll')
plt.legend()
plt.savefig('distribution_roll.png')

plt.figure()
plt.plot(data_train[:,4], label_train[:,4],'b*', label='Training Set')
plt.plot(data_test[:,4], label_test[:,4],'r*', label='Testing Set')
plt.xlabel('Ground Truth pitch')
plt.ylabel('Estimate pitch')
plt.legend()
plt.savefig('distribution_pitch.png')

plt.figure()
plt.plot(data_train[:,5], label_train[:,5],'b*', label='Training Set')
plt.plot(data_test[:,5], label_test[:,5],'r*', label='Testing Set')
plt.xlabel('Ground Truth yaw')
plt.ylabel('Estimate yaw')
plt.legend()
plt.savefig('distribution_yaw.png')

plt.show()