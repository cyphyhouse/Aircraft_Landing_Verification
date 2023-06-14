import numpy as np 
import matplotlib.pyplot as plt 
import os 

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, 'data/data_verif.txt')
label_path = os.path.join(script_dir, 'estimation_label/label_verif.txt')
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')
# table = [row.strip().split('\n') for row in x]

data_train = data[:, 1:]
label_train = label[:, 1:]
# env_params = data[:, 1]
# tmp = data[8000:, 1:]
# label_test = label[8000:, 1:]

# tmp = np.load(os.path.join(script_dir, 'ground_truth.npy'), allow_pickle=True)
# label_test = np.load(os.path.join(script_dir, 'estimation.npy'), allow_pickle=True)

# tmp = np.array(tmp)
# print(len(tmp))
# estimate_pos = np.array(estimate_pos).squeeze()

# data_test = []

# for i in range(len(tmp[:, 0])):
#     # print(tmp[i])
#     data_test.append(np.array([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]])) 
#     # print(state)
# data_test = np.array(data_test)

data_train_0 = np.empty((0,data_train.shape[1])) # 0.5-0.6
data_train_1 = np.empty((0,data_train.shape[1])) # 0.6-0.7
data_train_2 = np.empty((0,data_train.shape[1])) # 0.7-0.8
data_train_3 = np.empty((0,data_train.shape[1])) # 0.8-0.9
data_train_4 = np.empty((0,data_train.shape[1])) # 0.9-1.0
data_train_5 = np.empty((0,data_train.shape[1])) # 1.0-1.1
data_train_6 = np.empty((0,data_train.shape[1])) # 1.1-

label_train_0 = np.empty((0,label_train.shape[1])) # 0.5-0.6
label_train_1 = np.empty((0,label_train.shape[1])) # 0.6-0.7
label_train_2 = np.empty((0,label_train.shape[1])) # 0.7-0.8
label_train_3 = np.empty((0,label_train.shape[1])) # 0.8-0.9
label_train_4 = np.empty((0,label_train.shape[1])) # 0.9-1.0
label_train_5 = np.empty((0,label_train.shape[1])) # 1.0-1.1
label_train_6 = np.empty((0,label_train.shape[1])) # 1.1-

for i in range(data_train.shape[0]):
    if data_train[i,6]>=0.5 and data_train[i,6] < 0.6:
        data_train_0 = np.vstack((data_train_0, data_train[i,:]))
        label_train_0 = np.vstack((label_train_0, label_train[i,:]))
    elif data_train[i,6]>=0.6 and data_train[i,6] < 0.7:
        data_train_1 = np.vstack((data_train_1, data_train[i,:]))
        label_train_1 = np.vstack((label_train_1, label_train[i,:]))
    elif data_train[i,6]>=0.7 and data_train[i,6] < 0.8:
        data_train_2 = np.vstack((data_train_2, data_train[i,:]))
        label_train_2 = np.vstack((label_train_2, label_train[i,:]))
    elif data_train[i,6]>=0.8 and data_train[i,6] < 0.9:
        data_train_3 = np.vstack((data_train_3, data_train[i,:]))
        label_train_3 = np.vstack((label_train_3, label_train[i,:]))
    elif data_train[i,6]>=0.9 and data_train[i,6] < 1.0:
        data_train_4 = np.vstack((data_train_4, data_train[i,:]))
        label_train_4 = np.vstack((label_train_4, label_train[i,:]))
    elif data_train[i,6]>=1.0 and data_train[i,6] < 1.1:
        data_train_5 = np.vstack((data_train_5, data_train[i,:]))
        label_train_5 = np.vstack((label_train_5, label_train[i,:]))
    elif data_train[i,6]>=1.1:
        data_train_6 = np.vstack((data_train_6, data_train[i,:]))
        label_train_6 = np.vstack((label_train_6, label_train[i,:]))
    else:
        raise ValueError("Value out of bound")

# idx_list = np.where(data_train[:,0]<-3150)
# print(idx_list)
# plt.figure()
# plt.plot(data_train[idx_list,0],label_train[idx_list,0], 'b*')
# plt.show()

font = {'family': 'serif', 'weight': 'normal', 'size': 14}

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,0], label_train[:,0],'r*', label='Testing Set')
plt.plot(data_train_0[:,0], label_train_0[:,0], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,0], label_train_1[:,0], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,0], label_train_2[:,0], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,0], label_train_3[:,0], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,0], label_train_4[:,0], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,0], label_train_5[:,0], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,0], label_train_6[:,0], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth x', fontdict=font)
plt.ylabel('Estimate x', fontdict=font)
# plt.plot(data_train_1[:,0], abs(label_train_1[:,0] - data_train_1[:,0]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error x')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_x_test_by_lighting.png')

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,1], label_train[:,1],'r*', label='Testing Set')
plt.plot(data_train_0[:,1], label_train_0[:,1], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,1], label_train_1[:,1], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,1], label_train_2[:,1], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,1], label_train_3[:,1], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,1], label_train_4[:,1], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,1], label_train_5[:,1], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,1], label_train_6[:,1], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth y', fontdict=font)
plt.ylabel('Estimate y', fontdict=font)
# plt.plot(data_train_0[:,0], abs(label_train_0[:,1] - data_train_0[:,1]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error y')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_y_test_by_lighting.png')

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,2], label_train[:,2],'r*', label='Testing Set')
plt.plot(data_train_0[:,2], label_train_0[:,2], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,2], label_train_1[:,2], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,2], label_train_2[:,2], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,2], label_train_3[:,2], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,2], label_train_4[:,2], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,2], label_train_5[:,2], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,2], label_train_6[:,2], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth z', fontdict=font)
plt.ylabel('Estimate z', fontdict=font)
# plt.plot(data_train_0[:,0], abs(label_train_0[:,2] - data_train_0[:,2]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error z')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_z_test_by_lighting.png')

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,3], label_train[:,3],'r*', label='Testing Set')
plt.plot(data_train_0[:,3], label_train_0[:,3], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,3], label_train_1[:,3], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,3], label_train_2[:,3], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,3], label_train_3[:,3], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,3], label_train_4[:,3], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,3], label_train_5[:,3], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,3], label_train_6[:,3], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth roll', fontdict=font)
plt.ylabel('Estimate roll', fontdict=font)
# plt.plot(data_train_0[:,0], abs(label_train_0[:,3] - data_train_0[:,3]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error roll')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_roll_test_by_lighting.png')

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,4], label_train[:,4],'r*', label='Testing Set')
plt.plot(data_train_0[:,4], label_train_0[:,4], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,4], label_train_1[:,4], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,4], label_train_2[:,4], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,4], label_train_3[:,4], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,4], label_train_4[:,4], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,4], label_train_5[:,4], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,4], label_train_6[:,4], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth pitch', fontdict=font)
plt.ylabel('Estimate pitch', fontdict=font)
# plt.plot(data_train_0[:,0], abs(label_train_0[:,4] - data_train_0[:,4]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error pitch')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_pitch_test_by_lighting.png')

plt.figure()
# plt.plot(data_train[:,0], label_train[:,0],'b*', label='Training Set')
# plt.plot(data_train[:,5], label_train[:,5],'r*', label='Testing Set')
plt.plot(data_train_0[:,5], label_train_0[:,5], '*', color='tab:blue', label='0.5<=l<0.6')
plt.plot(data_train_1[:,5], label_train_1[:,5], '*', color='tab:orange', label='0.6<=l<0.7')
plt.plot(data_train_2[:,5], label_train_2[:,5], '*', color='tab:green', label='0.7<=l<0.8')
plt.plot(data_train_3[:,5], label_train_3[:,5], '*', color='tab:red', label='0.8<=l<0.9')
plt.plot(data_train_4[:,5], label_train_4[:,5], '*', color='tab:purple', label='0.9<=l<1.0')
plt.plot(data_train_5[:,5], label_train_5[:,5], '*', color='tab:brown', label='1.0<=l<1.1')
plt.plot(data_train_6[:,5], label_train_6[:,5], '*', color='tab:pink', label='1.1<=l')
plt.xlabel('Ground Truth yaw', fontdict=font)
plt.ylabel('Estimate yaw', fontdict=font)
# plt.plot(data_train_0[:,0], abs(label_train_0[:,5] - data_train_0[:,5]), 'b*', label='Error')
# plt.xlabel('Groubnd truth x')
# plt.ylabel('Error yaw')
plt.legend(prop={'family': 'serif'})
plt.savefig('distribution_yaw_test_by_lighting.png')

# plt.figure()
# plt.plot(data_train[:,1], label_train[:,1],'b*', label='Training Set')
# plt.plot(data_test[:,1], label_test[:,1],'r*', label='Testing Set')
# plt.xlabel('Ground Truth y')
# plt.ylabel('Estimate y')
# plt.legend()
# plt.savefig('distribution_y-vs-error_x_6.png')

# plt.figure()
# plt.plot(data_train[:,2], label_train[:,2],'b*', label='Training Set')
# plt.plot(data_test[:,2], label_test[:,2],'r*', label='Testing Set')
# plt.xlabel('Ground Truth z')
# plt.ylabel('Estimate z')
# plt.legend()
# plt.savefig('distribution_z-vs-error_x_6.png')

# plt.figure()
# plt.plot(data_train[:,3], label_train[:,3],'b*', label='Training Set')
# plt.plot(data_test[:,3], label_test[:,3],'r*', label='Testing Set')
# plt.xlabel('Ground Truth roll')
# plt.ylabel('Estimate roll')
# plt.legend()
# plt.savefig('distribution_roll-vs-error_x_6.png')

# plt.figure()
# plt.plot(data_train[:,4], label_train[:,4],'b*', label='Training Set')
# plt.plot(data_test[:,4], label_test[:,4],'r*', label='Testing Set')
# plt.xlabel('Ground Truth pitch')
# plt.ylabel('Estimate pitch')
# plt.legend()
# plt.savefig('distribution_pitch-vs-error_x_6.png')

# plt.figure()
# plt.plot(data_train[:,5], label_train[:,5],'b*', label='Training Set')
# plt.plot(data_test[:,5], label_test[:,5],'r*', label='Testing Set')
# plt.xlabel('Ground Truth yaw')
# plt.ylabel('Estimate yaw')
# plt.legend()
# plt.savefig('distribution_yaw-vs-error_x_6.png')

plt.show()