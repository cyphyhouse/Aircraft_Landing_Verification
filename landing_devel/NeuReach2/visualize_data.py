import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, 'data.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)

Er_list = []
r_list = []
r_x_list = []
r_y_list = []
r_z_list = []
r_roll_list = []
r_pitch_list = []
r_yaw_list = []
for i in range(len(data)):
    X0 = data[i][0]
    x, Ec, Er = X0 
    Er_list.append(Er[0])
    traces = data[i][1]
    traces = np.reshape(traces,(-1,6))
    max_r = 0
    max_r_x = 0
    max_r_y = 0
    max_r_z = 0
    max_r_roll = 0
    max_r_pitch = 0
    max_r_yaw = 0
    trace_mean = np.mean(traces, axis=0)
    for j in range(traces.shape[0]):
        r = np.linalg.norm(traces[j,:]-trace_mean)
        trace_diff = np.abs(traces[j,:]-trace_mean)
        max_r = max(r, max_r)
        max_r_x = max(trace_diff[0], max_r_x)
        max_r_y = max(trace_diff[1], max_r_y)
        max_r_z = max(trace_diff[2], max_r_z)
        max_r_roll = max(trace_diff[3], max_r_roll)
        max_r_pitch = max(trace_diff[4], max_r_pitch)
        max_r_yaw = max(trace_diff[5], max_r_yaw)
    r_list.append(max_r)
    r_x_list.append(max_r_x)
    r_y_list.append(max_r_y)
    r_z_list.append(max_r_z)
    r_roll_list.append(max_r_roll)
    r_pitch_list.append(max_r_pitch)
    r_yaw_list.append(max_r_yaw)

bin = []
bin_lb_list, bin_step = np.linspace(0, 0.35, 51, retstep=True)
Er_list = np.array(Er_list)
r_list = np.array(r_list)
r_x_list = np.array(r_x_list)
r_y_list = np.array(r_y_list)
r_z_list = np.array(r_z_list)
r_roll_list = np.array(r_roll_list)
r_pitch_list = np.array(r_pitch_list)
r_yaw_list = np.array(r_yaw_list)
Er_center_list = []
bound_list = []
bound_x_list = []
bound_y_list = []
bound_z_list = []
bound_roll_list = []
bound_pitch_list = []
bound_yaw_list = []
for i in range(len(bin_lb_list)-1):
    bin_lb = bin_lb_list[i]
    bin_ub = bin_lb + bin_step 
    idx = np.where((bin_lb<Er_list) & (Er_list<bin_ub))[0]
    Er_center_list.append((bin_lb+bin_ub)/2)
    r_sub_list = r_list[idx]
    percentile = np.percentile(r_sub_list, 80)
    bound_list.append(percentile)

    r_x_sub_list = r_x_list[idx]
    percentile_x = np.percentile(r_x_sub_list, 80)
    bound_x_list.append(percentile_x)
    r_y_sub_list = r_y_list[idx]
    percentile_y = np.percentile(r_y_sub_list, 80)
    bound_y_list.append(percentile_y)
    r_z_sub_list = r_z_list[idx]
    percentile_z = np.percentile(r_z_sub_list, 80)
    bound_z_list.append(percentile_z)
    r_roll_sub_list = r_roll_list[idx]
    percentile_roll = np.percentile(r_roll_sub_list, 80)
    bound_roll_list.append(percentile_roll)
    r_pitch_sub_list = r_pitch_list[idx]
    percentile_pitch = np.percentile(r_pitch_sub_list, 80)
    bound_pitch_list.append(percentile_pitch)
    r_yaw_sub_list = r_yaw_list[idx]
    percentile_yaw = np.percentile(r_yaw_sub_list, 80)
    bound_yaw_list.append(percentile_yaw)


plt.figure(0)
plt.plot(Er_center_list, bound_list, '*')

plt.figure(1)
plt.plot(Er_center_list, bound_x_list)
plt.figure(2)
plt.plot(Er_center_list, bound_y_list)
plt.figure(3)
plt.plot(Er_center_list, bound_z_list)
plt.figure(4)
plt.plot(Er_center_list, bound_roll_list)
plt.figure(5)
plt.plot(Er_center_list, bound_pitch_list)
plt.figure(6)
plt.plot(Er_center_list, bound_yaw_list)

plt.show()

# plt.plot(Er_list, r_list, '*')
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x0 = []
# e = []
# for i in range(len(data)):
#     traces = data[i][1]
#     tmp = np.reshape(traces,(-1,6))
#     tmp = np.mean(tmp, axis=0)
#     # if np.linalg.norm(tmp)>1e5:
#     #     continue
#     e.append(tmp)
#     X0 = data[i][0]
#     x, Ec, Er = X0 
#     x0.append(x)

# x0 = np.array(x0)
# e = np.array(e)
# print(np.max(np.abs(x0[:,0])),np.max(np.abs(x0[:,1])),np.max(np.abs(x0[:,2])))
# ax.scatter(x0[:,0], x0[:,1], x0[:,2])
# # ax.scatter(e[:,0], e[:,1], e[:,2])
# plt.show()