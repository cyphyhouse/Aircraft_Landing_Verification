import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

with open('vcs_sim_new.pickle','rb') as f:
    traj_list = pickle.load(f)
with open('vcs_estimate.pickle','rb') as f:
    estimate_traj_list = pickle.load(f)

# for i in range(len(traj_list)):
i=0
traj = np.array(traj_list[i])
est_traj = np.array(estimate_traj_list[i])

plt.figure(0)
plt.plot(traj[:,0], traj[:,1],'b')
plt.plot(traj[:-1,0], est_traj[:,0],'r')
plt.figure(1)
plt.plot(traj[:,0], traj[:,2],'b')
plt.plot(traj[:-1,0], est_traj[:,1],'r')
plt.figure(2)
plt.plot(traj[:,0], traj[:,3],'b')
plt.plot(traj[:-1,0], est_traj[:,2],'r')
plt.figure(3)
plt.plot(traj[:,0], traj[:,4],'b')
plt.plot(traj[:-1,0], est_traj[:,3],'r')
plt.figure(4)
plt.plot(traj[:,0], traj[:,5],'b')
plt.plot(traj[:-1,0], est_traj[:,4],'r')

plt.show()