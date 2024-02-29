import pickle 
import matplotlib.pyplot as plt 
import os 
from fixed_wing13_rain_snow import pre_process_data, apply_model_batch, remove_data, computeContract
import numpy as np 
from functools import reduce 

script_dir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(script_dir, 'exp1_res_safe_rain_snow.pickle')
with open(fn,'rb') as f:
    M, E, C = pickle.load(f)

fn = os.path.join(script_dir, '../data_train_exp1_rain_snow.pickle')
with open(fn, 'rb') as f:
    data = pickle.load(f)

data = pre_process_data(data)
data_removed = remove_data(data, E)
state_array, trace_array, E_array = data_removed 

M2 = computeContract(data, E)

plt.figure(0)
plt.scatter(state_array[:,0], trace_array[:,0])
c,r = apply_model_batch(M[0], state_array)
plt.scatter(state_array[:,0], c-r, color = 'r')
plt.scatter(state_array[:,0], c+r, color='r')
tmp0 = np.where((trace_array[:,0]<c-r) | (trace_array[:,0]>c+r))[0]

plt.figure(1)
plt.scatter(state_array[:,1], trace_array[:,1])
c,r = apply_model_batch(M[1], state_array)
plt.scatter(state_array[:,1], c-r, color = 'r')
plt.scatter(state_array[:,1], c+r, color='r')
tmp1 = np.where((trace_array[:,1]<c-r) | (trace_array[:,1]>c+r))[0]

plt.figure(2)
plt.scatter(state_array[:,2], trace_array[:,2])
c,r = apply_model_batch(M[2], state_array)
plt.scatter(state_array[:,2], c-np.abs(r), color = 'r')
plt.scatter(state_array[:,2], c+np.abs(r), color='r')
tmp2 = np.where((trace_array[:,2]<c-np.abs(r)) | (trace_array[:,2]>c+np.abs(r)))[0]

plt.figure(3)
plt.scatter(state_array[:,3], trace_array[:,3])
c,r = apply_model_batch(M[3], state_array)
plt.scatter(state_array[:,3], c-r, color = 'r')
plt.scatter(state_array[:,3], c+r, color='r')
tmp3 = np.where((trace_array[:,3]<c-r) | (trace_array[:,3]>c+r))[0]

plt.figure(4)
plt.scatter(state_array[:,4], trace_array[:,4])
c,r = apply_model_batch(M[4], state_array)
plt.scatter(state_array[:,4], c-r, color = 'r')
plt.scatter(state_array[:,4], c+r, color='r')
tmp4 = np.where((trace_array[:,4]<c-r) | (trace_array[:,4]>c+r))[0]

res = reduce(np.union1d, (tmp0, tmp1, tmp2, tmp3, tmp4))

print(len(res)/state_array.shape[0])

plt.show()