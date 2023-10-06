import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

def apply_model_batch(model, point):
    dim = model['dim']
    cc = model['coef_center']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[:,0]
    elif dim == 'y':
        point = point[:,(0,1)]
    elif dim == 'z':
        point = point
    elif dim == 'yaw':
        point = point[:,(0,3)]
    elif dim == 'pitch':
        point = point[:,(0,4)]

    if dim == 'x':
        x = point 
        center = cc[0]*x+cc[1]
        radius = cr[0]+x*cr[1]+x**2*cr[2]
        return center, radius
    elif dim == 'pitch' or dim == 'z':
        x = point[:,0]
        y = point[:,1]
        center = cc[0]*x + cc[1]*y +cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2]
        return center, radius
    else:
        x = point[:,0]
        y = point[:,1]
        center = cc[0]*x+cc[1]*y+cc[2]
        radius = cr[0]+x*cr[1]+y*cr[2]+x*y*cr[3]+x**2*cr[4]+y**2*cr[5]
        return center, radius

script_dir = os.path.dirname(os.path.realpath(__file__))
model_z_fn = os.path.join(script_dir, './verse/test.pickle')
with open(model_z_fn, 'rb') as f:
    M_out, E_out, _ = pickle.load(f)
    model_z = M_out[2]

data_file_path = os.path.join(script_dir, './data_train5.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)

state_list = []
Er_list = []
Ec_list = []
trace_list = []
e_list = []
for i in range(len(data)):
    X0 = data[i][0]
    x, Ec, Er = X0
    state_list.append(x)
    Ec_list.append(Ec) 
    Er_list.append(Er)
    traces = data[i][1]
    traces = np.reshape(traces,(-1,6))
    trace_list.append(traces)
    tmp = data[i][2]
    e = np.zeros((traces.shape[0],2))
    for j in range(traces.shape[0]):
        e[j,:] = tmp[j][1]
    e_list.append(e)

state_list = np.array(state_list)
Er_list = np.array(Er_list)
Ec_list = np.array(Ec_list)
trace_list = np.array(trace_list)
e_list = np.array(e_list)

for i in range(state_list.shape[0]):
    for env in E_out:
        # for j in range(trace_list.shape[1]):
        trace_seg = trace_list[i,:,2]
        state_seg = np.zeros(trace_seg.shape[0])
        state_seg[:] = state_list[i,2]
        tmp_seg = np.zeros(trace_seg.shape[0])
        tmp_seg[:] = state_list[i,0]
        e1_seg = e_list[i,:,0]
        e2_seg = e_list[i,:,1]
        idx = np.where(
            (e2_seg>env[0,1]) & \
            (e2_seg<env[1,1]) & \
            (e1_seg>env[0,0]) & \
            (e1_seg<env[1,0]))
        if idx[0].size == 0:
            continue
        trace_seg = trace_seg[idx]
        state_seg = state_seg[idx]
        tmp_seg = tmp_seg[idx]
        plt.plot(state_seg, trace_seg, 'b*')
        tmp_data = np.hstack((tmp_seg.reshape((-1,1)), state_seg.reshape((-1,1))))
        c, r = apply_model_batch(model_z, tmp_data)
        plt.plot(state_seg, c-r, 'g*')
        plt.plot(state_seg, c+r, 'g*')

data_fn = os.path.join(script_dir, './verse/vcs_sim.pickle')
with open(data_fn, 'rb') as f:
    data = pickle.load(f)

estm_fn = os.path.join(script_dir, './verse/vcs_estimate.pickle')
with open(estm_fn, 'rb') as f:
    estm = pickle.load(f)

data_z = np.array(data)[:,:-1,3].reshape((-1,1))
data_x = np.array(data)[:,:-1,1].reshape((-1,1))
data  = np.hstack((data_x, data_z))
estm_z = np.array(estm)[:,:,2].flatten()
# plt.plot(data_z, estm_z, 'r*')
c, r = apply_model_batch(model_z, data)
plt.plot(data_z, c-r, 'g*')
plt.plot(data_z, c+r, 'g*')

plt.show()

