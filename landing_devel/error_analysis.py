import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
ground_truth_position = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/ground_truth.npy', allow_pickle=True)
estimate_pos = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation.npy', allow_pickle=True)

data_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data.txt'
label_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label.txt'
x = np.loadtxt(data_path, delimiter=',')
y = np.loadtxt(label_path, delimiter=',')

ground_truth_position = x[8000:, 1:]
estimate_pos = y[8000:, 1:]

# ground_truth_position = np.array(ground_truth_position)
# print(len(ground_truth_position))
# estimate_pos = np.array(estimate_pos).squeeze()

# # fields = ['X Estimation Error', 'Y Estimation Error', 'Z Estimation Error', 'Roll Estimation Error', 'Pitch Estimation Error', 'Yaw Estimation Error']
# fields = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
# # data rows of csv file
# rows = []
 
# # name of csv file
# filename = "error_analysis_ground_truth.csv"
 

# for i in range(len(ground_truth_position[:, 0])):
#     # error = []
#     ground_truth = []
#     for j in range(len(ground_truth_position[0, :])):
#         # error.append(ground_truth_position[i, j]-estimate_pos[i, j])
#         ground_truth.append(ground_truth_position[i, j])
#     rows.append(ground_truth)
    
# # writing to csv file
# with open(filename, 'w') as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)
     
#     # writing the fields
#     csvwriter.writerow(fields)
     
#     # writing the data rows
#     csvwriter.writerows(rows)

# --------------------------------------------------------------------------------------------------------------------------------------
# Surrogate Model error analysis
estimated_state_surrogate = []
ground_truth_state = []
surrogate_model = tf.keras.models.load_model('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/surrogate_model/model.keras')
for i in range(len(ground_truth_position[:, 0])):
    # print(ground_truth_position[i])
    ground_truth_state.append(np.array([ground_truth_position[i][0], ground_truth_position[i][1], ground_truth_position[i][2], ground_truth_position[i][3], ground_truth_position[i][4], ground_truth_position[i][5]])) 
    # print(state)
ground_truth_state = np.array(ground_truth_state)

estimated_state_surrogate.extend(surrogate_model.predict(ground_truth_state))
# estimated_state_surrogate.extend(surrogate_model.predict(ground_truth_state[1000:2000, :]))
# estimated_state_surrogate.extend(surrogate_model.predict(ground_truth_state[2000:3000, :]))
# estimated_state_surrogate.extend(surrogate_model.predict(ground_truth_state[3000:4000, :]))
# estimated_state_surrogate.extend(surrogate_model.predict(ground_truth_state[4000:, :]))
estimated_state_surrogate = np.array(estimated_state_surrogate)
# print(estimated_state_surrogate)
font = {'family': 'serif', 'weight': 'normal', 'size': 12}

plt.figure()
plt.plot(ground_truth_state[:,0], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,0], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,0], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('X coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_x.png', dpi=300)


plt.figure()
plt.plot(ground_truth_state[:,1], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,1], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,1], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Y coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_y.png', dpi=300)

plt.figure()
plt.plot(ground_truth_state[:,2], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,2], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,2], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Z coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_z.png', dpi=300)

plt.figure()
plt.plot(ground_truth_state[:,3], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,3], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,3], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Roll angle (rad)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_roll.png', dpi=300)

plt.figure()
plt.plot(ground_truth_state[:,4], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,4], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,4], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Pitch angle (rad)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_pitch.png', dpi=300)

plt.figure()
plt.plot(ground_truth_state[:,5], 'g', label='Ground Truth')
plt.plot(estimated_state_surrogate[:,5], 'r', label='Surrogate Estimation')
plt.plot(estimate_pos[:,5], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Yaw angle (rad)', fontdict=font)
plt.legend()
plt.savefig('surrogate_estimation_comparison_yaw.png', dpi=300)


# plt.figure()
# plt.plot(ground_truth_state[:,0], np.abs((estimated_state_surrogate[:,0] - ground_truth_state[:,0])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('X estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_x_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_state[:,1], np.abs((estimated_state_surrogate[:,1] - ground_truth_state[:,1])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Y estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_y_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_state[:,2], np.abs((estimated_state_surrogate[:,2] - ground_truth_state[:,2])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Z estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_z_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_state[:,3], np.abs((estimated_state_surrogate[:,3] - ground_truth_state[:,3])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Roll estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_roll_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_state[:,4], np.abs((estimated_state_surrogate[:,4] - ground_truth_state[:,4])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Pitch estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_pitch_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_state[:,5], np.abs((estimated_state_surrogate[:,5] - ground_truth_state[:,5])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Yaw estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('surrogate_estimation_yaw_error.png', dpi=300)

plt.show()