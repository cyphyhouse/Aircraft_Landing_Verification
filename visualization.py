import numpy as np 
import matplotlib.pyplot as plt 


ground_truth_position = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/ground_truth.npy', allow_pickle=True)
estimate_pos = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation.npy', allow_pickle=True)
# ref_states = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/ref_states.npy', allow_pickle=True)
# averaged_states = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/averaged_states.npy', allow_pickle=True)
# nominal_states = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/nominal_states.npy', allow_pickle=True)
print(len(ground_truth_position), len(estimate_pos))

ground_truth_position = np.array(ground_truth_position)
estimate_pos = np.array(estimate_pos).squeeze()
# ref_states = np.array(ref_states)
ground_truth_position[:, 1] = -ground_truth_position[:, 1]
ground_truth_position[:, 2] = -ground_truth_position[:, 2]

estimate_pos[:, 1] = -estimate_pos[:, 1]
estimate_pos[:, 2] = -estimate_pos[:, 2]

for i in range(len(ground_truth_position[:, 0])):
    ground_truth_position[i, 3] = np.rad2deg(ground_truth_position[i, 3])
    ground_truth_position[i, 4] = np.rad2deg(ground_truth_position[i, 4])
    ground_truth_position[i, 5] = np.rad2deg(ground_truth_position[i, 5])
    estimate_pos[i, 3] = np.rad2deg(estimate_pos[i, 3])
    estimate_pos[i, 4] = np.rad2deg(estimate_pos[i, 4])
    estimate_pos[i, 5] = np.rad2deg(estimate_pos[i, 5])

# ref_states[:, 1] = -ref_states[:, 1]
# ref_states[:, 2] = -ref_states[:, 2]

# nominal_states[:, 1] = -nominal_states[:, 1]
# nominal_states[:, 2] = -nominal_states[:, 2]
font = {'family': 'serif', 'weight': 'normal', 'size': 12}

# plt.figure()
# plt.plot(ground_truth_position[20:,0], 'g', label='Ground Truth')
# plt.plot(estimate_pos[20:,0], 'b', label='Estimation')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('X coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_comparison_x.png', dpi=300)

# font = {'family': 'serif', 'weight': 'normal', 'size': 12}
# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,0] - ground_truth_position[20:,0]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('X estimation error (m)', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('estimation_x_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(estimate_pos[20:,1], 'b', label='Estimation')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_comparison_y.png', dpi=300)

# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,1] - ground_truth_position[20:,1]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y estimation error (m)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_y_error.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,2], 'g', label='Ground Truth')
# plt.plot(estimate_pos[20:,2], 'b', label='Estimation')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Z coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_comparison_z.png', dpi=300)

# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,2] - ground_truth_position[20:,2]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Z estimation error (m)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_z_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[20:,5], 'g', label='Ground Truth')
plt.plot(estimate_pos[20:,5], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Yaw (deg)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_yaw.png', dpi=300)

# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,5] - ground_truth_position[20:,5]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Yaw estimation error (deg)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_yaw_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[20:,3], 'g', label='Ground Truth')
plt.plot(estimate_pos[20:,3], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Roll (deg)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_roll.png', dpi=300)

# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,3] - ground_truth_position[20:,3]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Roll estimation error (deg)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_roll_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[20:,4], 'g', label='Ground Truth')
plt.plot(estimate_pos[20:,4], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Pitch (deg)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_pitch.png', dpi=300)

# plt.figure()
# plt.plot(np.abs(estimate_pos[20:,4] - ground_truth_position[20:,4]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Pitch estimation error (deg)', fontdict=font)
# plt.legend()
# plt.savefig('estimation_pitch_error.png', dpi=300)

'''
Percentage error.
'''
# font = {'family': 'serif', 'weight': 'normal', 'size': 12}
# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,0] - ground_truth_position[20:,0])/ground_truth_position[20:,0]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('X estimation error percentage', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('estimation_x_error_percentage.png', dpi=300)

# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,1] - ground_truth_position[20:,1])/ground_truth_position[20:,1]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y estimation error percentage', fontdict=font)
# plt.legend()
# plt.savefig('estimation_y_error_percentage.png', dpi=300)

# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,2] - ground_truth_position[20:,2])/ground_truth_position[20:,2]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Z estimation error percentage', fontdict=font)
# plt.legend()
# plt.savefig('estimation_z_error_percentage.png', dpi=300)


# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,5] - ground_truth_position[20:,5])/ground_truth_position[20:,5]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Yaw estimation error percentage', fontdict=font)
# plt.legend()
# plt.savefig('estimation_yaw_error_percentage.png', dpi=300)

# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,3] - ground_truth_position[20:,3])/(ground_truth_position[20:,3]+1)), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Roll estimation error percentage', fontdict=font)
# plt.legend()
# plt.savefig('estimation_roll_error_percentage.png', dpi=300)


# plt.figure()
# plt.plot(np.abs((estimate_pos[20:,4] - ground_truth_position[20:,4])/ground_truth_position[20:,4]), label='Errors')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Pitch estimation error percentage', fontdict=font)
# plt.legend()
# plt.savefig('estimation_pitch_error_percentage.png', dpi=300)



# '''
# Error VS Ground Truth.
# '''
# font = {'family': 'serif', 'weight': 'normal', 'size': 12}
# plt.figure()
# plt.plot(ground_truth_position[20:,0], np.abs((estimate_pos[20:,0] - ground_truth_position[20:,0])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('X estimation error VS Ground Truth', fontdict=font)
# plt.legend(prop={'family': 'serif'})
# plt.savefig('estimation_x_error_VS_ground_truth.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], np.abs((estimate_pos[20:,1] - ground_truth_position[20:,1])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Y estimation error VS Ground Truth', fontdict=font)
# plt.legend()
# plt.savefig('estimation_y_error_VS_ground_truth.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,2], np.abs((estimate_pos[20:,2] - ground_truth_position[20:,2])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Z estimation error VS Ground Truth', fontdict=font)
# plt.legend()
# plt.savefig('estimation_z_error_VS_ground_truth.png', dpi=300)


# plt.figure()
# plt.plot(ground_truth_position[20:,5], np.abs((estimate_pos[20:,5] - ground_truth_position[20:,5])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Yaw estimation error VS Ground Truth', fontdict=font)
# plt.legend()
# plt.savefig('estimation_yaw_error_VS_ground_truth.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,3], np.abs((estimate_pos[20:,3] - ground_truth_position[20:,3])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Roll estimation error VS Ground Truth', fontdict=font)
# plt.legend()
# plt.savefig('estimation_roll_error_VS_ground_truth.png', dpi=300)


# plt.figure()
# plt.plot(ground_truth_position[20:,4], np.abs((estimate_pos[20:,4] - ground_truth_position[20:,4])), label='Errors')
# plt.xlabel('Ground Truth', fontdict=font)
# plt.ylabel('Pitch estimation error VS Ground Truth', fontdict=font)
# plt.legend()
# plt.savefig('estimation_pitch_error_VS_ground_truth.png', dpi=300)





'''
Tracking error.
'''
# font = {'family': 'serif', 'weight': 'normal', 'size': 12}
# plt.figure()
# plt.plot(ground_truth_position[20:,0], 'g', label='Ground Truth')
# plt.plot(ref_states[20:,0], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('X coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_x.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(ref_states[20:,1], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_y.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,2], 'g', label='Ground Truth')
# plt.plot(ref_states[20:,2], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Z coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_z.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,5], 'g', label='Ground Truth')
# plt.plot(ref_states[20:,3], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Yaw coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_yaw.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(ref_states[20:,1], 'b', label='Reference')
# plt.plot(averaged_states, 'r', label='Averaged')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_Y.png', dpi=300)

###############################################################

# font = {'family': 'serif', 'weight': 'normal', 'size': 12}

# plt.figure()
# plt.plot(ground_truth_position[20:,0], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,0], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('X coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_x.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,1], 'b', label='Reference')
# plt.ylim([-30, 30])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_y.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,2], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,2], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Z coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_z.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,5], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,3], 'b', label='Reference')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Yaw coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_yaw.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,1], 'b', label='Reference')
# plt.plot(averaged_states, 'r', label='Averaged')
# # plt.ylim([])
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_Y.png', dpi=300)

# plt.figure()
# plt.plot(ground_truth_position[20:,0], ground_truth_position[20:,1], 'g', label='Ground Truth')
# plt.plot(nominal_states[20:,0], nominal_states[20:,1], 'b', label='Reference')
# plt.xlabel('Time step', fontdict=font)
# plt.ylabel('Y coordinate (m)', fontdict=font)
# plt.legend()
# plt.savefig('tracking_comparison_Y.png', dpi=300)


plt.show()