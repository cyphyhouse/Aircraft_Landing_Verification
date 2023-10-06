import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 
from fixed_wing12 import pre_process_data, refineEnv, computeContract, partitionE, remove_data

dim = 0

def apply_model_batch(model, point):
    dim = model['dim']
    cc = model['coef_center']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[:,0]
    elif dim == 'y':
        point = point[:,(0,1)]
    elif dim == 'z':
        point = point[:, (0,2)]
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

def get_bin_max_list(states, c, r):
    if dim == 1:
        bins = np.linspace(-100, 100, 100)
    elif dim == 2:
        bins = np.linspace(40, 130, 100)
    elif dim == 3:
        bins = np.linspace(-0.5, 0.5, 50)
    x = []
    y1 = [] 
    y2 = []
    for i in range(bins.shape[0]-1):
        bin_low = bins[i]
        bin_high = bins[i+1]
        idx_in_bin = np.where((bin_low< states[:,dim]) & (states[:,dim]<bin_high))[0]
        if idx_in_bin.size != 0:
            r_idx = np.argmax(r[idx_in_bin])
            x.append(states[idx_in_bin, dim][r_idx])
            y1.append(c[idx_in_bin][r_idx] - r[idx_in_bin][r_idx])
            y2.append(c[idx_in_bin][r_idx] + r[idx_in_bin][r_idx])
    return x, y1, y2

script_dir = os.path.dirname(os.path.realpath(__file__))
# model_z_fn = os.path.join(script_dir, './verse/test.pickle')
# with open(model_z_fn, 'rb') as f:
#     M_out, E_out, _ = pickle.load(f)
#     model_z = M_out[2]

E = np.array([
    [0.2, -0.1],
    [1.2, 0.6]
])

E = partitionE(E)

data_file_path = os.path.join(script_dir, '../data_train_exp1.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)
data = pre_process_data(data)
state_array, trace_array, E_array = data

data_file_path = os.path.join(script_dir, '../data_eval_exp1.pickle')
with open(data_file_path,'rb') as f:
    data_eval = pickle.load(f)
data_eval = pre_process_data(data_eval)
state_eval, trace_eval, E_eval = data_eval

fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
color_list = ['#80b1d3', '#ffed6f', '#fb8072']
labels = ['no refine', '3 refine', '6 refine']
for i in range(3):
    M = computeContract(data, E)    

        # for env in E_out:
        #     # for j in range(trace_list.shape[1]):
        #     trace_seg = trace_list[i,:,2]
        #     state_seg = np.zeros(trace_seg.shape[0])
        #     state_seg[:] = state_list[i,2]
        #     tmp_seg = np.zeros(trace_seg.shape[0])
        #     tmp_seg[:] = state_list[i,0]
        #     e1_seg = e_list[i,:,0]
        #     e2_seg = e_list[i,:,1]
        #     idx = np.where(
        #         (e2_seg>env[0,1]) & \
        #         (e2_seg<env[1,1]) & \
        #         (e1_seg>env[0,0]) & \
        #         (e1_seg<env[1,0]))
        #     if idx[0].size == 0:
        #         continue
        #     trace_seg = trace_seg[idx]
        #     state_seg = state_seg[idx]
        #     tmp_seg = tmp_seg[idx]
        #     plt.plot(state_seg, trace_seg, 'b*')
        #     tmp_data = np.hstack((tmp_seg.reshape((-1,1)), state_seg.reshape((-1,1))))
    # tmp_data = remove_data(data, E)
    # tmp_state, tmp_trace, tmp_E = tmp_data

    tmp_data = remove_data(data_eval, E)
    tmp_state, tmp_trace, tmp_E = tmp_data

    c1,r1 = apply_model_batch(M[0], tmp_state)
    c2,r2 = apply_model_batch(M[1], tmp_state)
    c3,r3 = apply_model_batch(M[2], tmp_state)
    c4,r4 = apply_model_batch(M[3], tmp_state)
    c5,r5 = apply_model_batch(M[4], tmp_state)
    idx = np.where(
        (c1-r1<tmp_trace[:,0]) & (tmp_trace[:,0]<c1+r1) &\
        (c2-r2<tmp_trace[:,1]) & (tmp_trace[:,1]<c2+r2) &\
        (c3-r3<tmp_trace[:,2]) & (tmp_trace[:,2]<c3+r3) &\
        (c4-r4<tmp_trace[:,3]) & (tmp_trace[:,3]<c4+r4) &\
        (c5-r5<tmp_trace[:,4]) & (tmp_trace[:,4]<c5+r5)
    )[0]
    print(">>>>> accuracy 1", len(idx)/tmp_trace.shape[0])

    c1,r1 = apply_model_batch(M[0], state_eval)
    c2,r2 = apply_model_batch(M[1], state_eval)
    c3,r3 = apply_model_batch(M[2], state_eval)
    c4,r4 = apply_model_batch(M[3], state_eval)
    c5,r5 = apply_model_batch(M[4], state_eval)
    idx = np.where(
        (c1-r1<trace_eval[:,0]) & (trace_eval[:,0]<c1+r1) &\
        (c2-r2<trace_eval[:,1]) & (trace_eval[:,1]<c2+r2) &\
        (c3-r3<trace_eval[:,2]) & (trace_eval[:,2]<c3+r3) &\
        (c4-r4<trace_eval[:,3]) & (trace_eval[:,3]<c4+r4) &\
        (c5-r5<trace_eval[:,4]) & (trace_eval[:,4]<c5+r5)
    )[0]
    print(">>>>> accuracy 2", len(idx)/trace_eval.shape[0])

    # c, r = apply_model_batch(M[dim], tmp_state)
    # ax.scatter(tmp_state[:,0], tmp_state[:,dim], c-r, 'g*')
    # ax.scatter(tmp_state[:,0], tmp_state[:,dim], c+r, 'g*')

    # tmp_state = tmp_state[:10,:]
    # if dim == 2:
    #     tmp_state[:,0] = -3000
    #     tmp_state[:,2] = np.linspace(40, 130, 10)
    c,r = apply_model_batch(M[dim], state_eval)
    # c = c.reshape(tmp_state.shape)
    # r = r.reshape(X.shape)
    
    # ax.plot_surface(X, Y, c-r, color = color_list[i])
    # ax.plot_surface(X, Y, c+r, color = color_list[i])
    # plt.plot(state_eval[:,dim], c-r, color_list[i]+"*")
    # plt.plot(state_eval[:,dim], c+r, color_list[i]+"*")
    if dim != 0:
        x, y1, y2 = get_bin_max_list(state_eval, c, r)
    else:
        x = np.linspace(-3000, -2000, 1000)
        model = M[0]
        cc = model['coef_center']
        cr = model['coef_radius']
        center = cc[0]*x+cc[1]
        radius = cr[0]+x*cr[1]+x**2*cr[2]
        y1 = center-radius 
        y2 = center+radius

    plt.plot(x, y1, color_list[i])
    plt.plot(x, y2, color_list[i])
    plt.fill_between(x, y1, y2, color = color_list[i], alpha = 0.5, label=labels[i])

    for i in range(3):
        E = refineEnv(E, M, data)

data_file_path = os.path.join(script_dir, '../data_eval_exp1.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)
data = pre_process_data(data)
# tmp_data = remove_data(data, E)
tmp_state, tmp_trace, tmp_E = data
# ax.scatter(tmp_state[:,0], tmp_state[:,dim], tmp_trace[:,dim], 'b*')
plt.scatter(tmp_state[:,dim], tmp_trace[:,dim], color = '#bc80bd', label='test data')

data_file_path = os.path.join(script_dir, '../data_grounding_exp1.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)
data = pre_process_data(data)
# tmp_data = remove_data(data, E)
tmp_state, tmp_trace, tmp_E = data
# ax.scatter(tmp_state[:,0], tmp_state[:,dim], tmp_trace[:,dim], 'b*')
plt.scatter(tmp_state[:,dim], tmp_trace[:,dim], color = '#8dd3c7', label='ground env')

if dim == 0:
    plt.xlabel('Ground Truth x', fontsize=16)
    plt.ylabel('Estimated x', fontsize=16)
elif dim == 1:
    plt.xlabel('Ground Truth y', fontsize=16)
    plt.ylabel('Estimated y', fontsize=16)
elif dim == 2:
    plt.xlabel('Ground Truth z', fontsize=16)
    plt.ylabel('Estimated z', fontsize=16)
elif dim == 3:
    plt.xlabel('Ground Truth $\psi$', fontsize=16)
    plt.ylabel('Estimated $\psi$', fontsize=16)
plt.legend(fontsize=16)
plt.show()
