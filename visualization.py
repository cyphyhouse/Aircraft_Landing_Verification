import numpy as np 
import matplotlib.pyplot as plt 


ground_truth_position = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/ground_truth.npy')
estimate_pos = np.load('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation.npy')

print(len(ground_truth_position), len(estimate_pos))

ground_truth_position = np.array(ground_truth_position)
estimate_pos = np.array(estimate_pos).squeeze()

# plt.figure()
# plt.plot(np.linalg.norm(ground_truth_position-estimate_pos, axis=1), label='Errors')
# plt.xlabel('Time step', fontsize=20)
# plt.ylabel('Estimation error', fontsize=20)
# plt.legend()
# plt.savefig('estimation_comparison_error.png')

font = {'family': 'serif', 'weight': 'normal', 'size': 12}

plt.figure()
plt.plot(ground_truth_position[100:,0], 'g', label='Ground Truth')
plt.plot(estimate_pos[100:,0], 'b', label='Estimation')
# plt.ylim([])
plt.xlabel('Time step', fontdict=font)
plt.ylabel('X coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_x.png', dpi=300)

font = {'family': 'serif', 'weight': 'normal', 'size': 12}
plt.figure()
plt.plot(np.abs(estimate_pos[100:,0] - ground_truth_position[100:,0]), label='Errors')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('X estimation error (m)', fontdict=font)
plt.legend(prop={'family': 'serif'})
plt.savefig('estimation_x_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[100:,1], 'g', label='Ground Truth')
plt.plot(estimate_pos[100:,1], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Y coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_y.png', dpi=300)

plt.figure()
plt.plot(np.abs(estimate_pos[100:,1] - ground_truth_position[100:,1]), label='Errors')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Y estimation error (m)', fontdict=font)
plt.legend()
plt.savefig('estimation_y_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[100:,2], 'g', label='Ground Truth')
plt.plot(estimate_pos[100:,2], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Z coordinate (m)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_z.png', dpi=300)

plt.figure()
plt.plot(np.abs(estimate_pos[100:,2] - ground_truth_position[100:,2]), label='Errors')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Z estimation error (m)', fontdict=font)
plt.legend()
plt.savefig('estimation_z_error.png', dpi=300)

plt.figure()
plt.plot(ground_truth_position[100:,5], 'g', label='Ground Truth')
plt.plot(estimate_pos[100:,5], 'b', label='Estimation')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Yaw (rad)', fontdict=font)
plt.legend()
plt.savefig('estimation_comparison_yaw.png', dpi=300)

plt.figure()
plt.plot(np.abs(estimate_pos[100:,5] - ground_truth_position[100:,5]), label='Errors')
plt.xlabel('Time step', fontdict=font)
plt.ylabel('Yaw estimation error (rad)', fontdict=font)
plt.legend()
plt.savefig('estimation_yaw_error.png', dpi=300)


# plt.figure()
# plt.plot(ground_truth_position[:,3], 'g', label='Ground Truth')
# plt.plot(estimate_pos[:,3], 'b', label='Estimation Truth')
# plt.xlabel('Time step', fontsize=20)
# plt.ylabel('roll', fontsize=20)
# plt.legend()
# plt.savefig('estimation_comparison_roll.png')

# plt.figure()
# plt.plot(ground_truth_position[:,3], np.abs(estimate_pos[:,3] - ground_truth_position[:,3]), label='Errors')
# plt.xlabel('Ground truth roll', fontsize=20)
# plt.ylabel('roll estimation error (rad)', fontsize=20)
# plt.legend()
# plt.savefig('estimation_roll_error.png')

# plt.figure()
# plt.plot(ground_truth_position[:,4], 'g', label='Ground Truth')
# plt.plot(estimate_pos[:,4], 'b', label='Estimation Truth')
# plt.xlabel('Time step', fontsize=20)
# plt.ylabel('pitch', fontsize=20)
# plt.legend()
# plt.savefig('estimation_comparison_pitch.png')

# plt.figure()
# plt.plot(ground_truth_position[:,4], np.abs(estimate_pos[:,4] - ground_truth_position[:,4]), label='Errors')
# plt.xlabel('Ground truth pitch', fontsize=20)
# plt.ylabel('pitch estimation error (rad)', fontsize=20)
# plt.legend()
# plt.savefig('estimation_pitch_error.png')

# plt.figure()
# plt.plot(ground_truth_position[:,5], 'g', label='Ground Truth')
# plt.plot(estimate_pos[:,5], 'b', label='Estimation Truth')
# plt.xlabel('Time step', fontsize=20)
# plt.ylabel('Yaw (rad)', fontsize=20)
# plt.legend()
# plt.savefig('estimation_comparison_yaw.png')

# plt.figure()
# plt.plot(ground_truth_position[:,5], np.abs(estimate_pos[:,5] - ground_truth_position[:,5]), label='Errors')
# plt.xlabel('Ground truth yaw', fontsize=20)
# plt.ylabel('yaw estimation error (rad)', fontsize=20)
# plt.legend()
# plt.savefig('estimation_yaw_error.png')

plt.show()