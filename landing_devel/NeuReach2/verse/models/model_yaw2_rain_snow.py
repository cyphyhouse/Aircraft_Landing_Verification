import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

dim = 3

def compute_model_yaw(state_array, trace_array, E_array, pc=0.9, pr=0.95):
    X_process = state_array[:,(0,dim)]
    Y_process = trace_array[:,dim]

    model_center = sm.QuantReg(Y_process, sm.add_constant(X_process))
    result = model_center.fit(q=0.5)
    cc = [result.params[1], result.params[2], result.params[0]]  

    X1 = state_array[:,0]
    X2 = state_array[:, dim]
    X_radius = np.hstack((
        X1.reshape((-1,1)),
        X2.reshape((-1,1)),
        (X1*X2).reshape((-1,1)),
        (X1**2).reshape((-1,1)),
        (X2**2).reshape((-1,1))
    ))
    Y_radius = np.abs(Y_process - (cc[2] + X_process[:,0]*cc[0] + X_process[:,1]*cc[1]))
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
    result = model_radius.fit(q=pr)
    cr = result.params

    res = {
        'dim': 'yaw',
        'coef_center':cc,
        'coef_radius': cr.tolist()
    }
    return res

# def compute_model_yaw(state_array, trace_array, E_array, pc=0.9, pr=0.95):
#     state_array_process = np.zeros((0,2))
#     trace_array_process = np.array([])
#     E_array_process = np.zeros((0,2))
#     Xp0 = np.arange(-3000, -2000, 200)
#     Xp1 = np.arange(-100, 100, 20)
#     Ep1 = np.arange(0.0, 1.0, 0.2)
#     Ep2 = np.arange(0.0,1.0,0.2)
#     total_num = 0
#     for i in range(Xp0.shape[0]):
#         for j in range(Xp1.shape[0]):
#             for k in range(Ep1.shape[0]):
#                 for l in range(Ep2.shape[0]):
#                     idx = np.where((state_array[:,0]>Xp0[i]) &\
#                                     (state_array[:,0]<(Xp0[i]+200)) &\
#                                     (state_array[:,dim]>Xp1[j]) &\
#                                     (state_array[:,dim]<(Xp1[j]+20)) &\
#                                     (E_array[:,0]>Ep1[k]) &\
#                                     (E_array[:,0]<(Ep1[k]+0.2)) &\
#                                     (E_array[:,1]>Ep2[l]) &\
#                                     (E_array[:,1]<(Ep2[l]+0.2)))[0]
#                     X_partition = state_array[idx,: ][:, (0, dim)]
#                     E_partition = E_array[idx,:]
#                     total_num += X_partition.size
#                     trace_partition = trace_array[idx,dim]
#                     tmp = np.abs(trace_partition - X_partition[:,1])
#                     if tmp.size!=0:
#                         percentile = np.percentile(tmp, pc*100)
#                         state_array_process = np.concatenate((state_array_process,X_partition[tmp<percentile]))
#                         E_array_process = np.vstack((E_array_process,E_partition[tmp<percentile]))
#                         trace_array_process = np.concatenate((trace_array_process,trace_partition[tmp<percentile]))
#     # print(total_num)
#     X_process = state_array_process.reshape((-1,2))
#     Y_process = trace_array_process

#     # fig6 = plt.figure(6)
#     # plt.plot(state_list_process, trace_mean_list_process, 'b.')

#     # fig5 = plt.figure(5)
#     # ax5 = fig5.add_subplot(projection='3d')
#     # ax5.scatter(state_list_process, Ec_list_process, trace_mean_list_process, 'b')

#     mcc = LinearRegression()
#     mcc.fit(X_process,Y_process)
#     # -------------------------------------

#     # Getting Model for Radius
#     X = state_array[:, (0,dim)]
#     X1 = state_array[:,0]
#     X2 = state_array[:, dim]
#     center_center = mcc.predict(X)
#     trace_array_radius = trace_array[:,(0, dim)]

#     tmp = np.array(tmp)
#     Y_radius = np.abs(trace_array_radius[:,1]-center_center)
#     # Y_radius = np.abs(trace_list_radius[:,0]-X_radius)
#     quantile = pr
#     X_radius = np.hstack((
#         X1.reshape((-1,1)),
#         X2.reshape((-1,1)),
#         (X1*X2).reshape((-1,1)),
#         (X1**2).reshape((-1,1)),
#         (X2**2).reshape((-1,1))
#     ))
#     model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
#     result = model_radius.fit(q=quantile)
#     cr = result.params

#     cc = mcc.coef_.tolist()+[mcc.intercept_]
#     min_cc = 0
#     min_r = 0
#     for i in range(state_array.shape[0]):
#         x = state_array[i,0]
#         y = state_array[i, dim]
#         center_center = cc[0]*x + cc[1]*y + cc[2]
#         if center_center < min_cc:
#             min_cc = center_center
#         radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
#         if radius < min_r:
#             min_r = radius
#     cr[0] += (-min_r)
#     res = {
#         'dim': 'yaw',
#         'coef_center':cc,
#         'coef_radius': cr.tolist()
#     }
#     return res
# -------------------------------------

# Testing the obtained models
# The center of perception contract. 
# mcc # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of possible center 
# model_center_radius 
# ccr # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of perception contract 
# model_radius 
# cr # Input to this function is the ground truth state

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../../data_train3.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    trace_list = []
    E_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        init = data[i][2]
        for tmp in init:
            E_list.append(tmp[1])
    # Getting Model for center center model
    state_list = np.array(state_list)
    E_list = np.array(E_list)

    # Flatten Lists
    state_array = np.zeros((E_list.shape[0],state_list.shape[1]))
    trace_array = np.zeros(state_array.shape)
    E_array = E_list

    num = trace_list[0].shape[0]
    for i in range(state_list.shape[0]):
        state_array[i*num:(i+1)*num,:] = state_list[i,:]
        trace_array[i*num:(i+1)*num,:] = trace_list[i] 

    res = compute_model_yaw(state_array, trace_array, E_array, pc = 0.8, pr=0.99)
    cc = res['coef_center']
    cr = res['coef_radius']
    # with open(os.path.join(script_dir,'model_yaw2.json'),'w+') as f:
    #     json.dump(res, f, indent=4)

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i, dim]
        center_center = cc[0]*x + cc[1]*y + cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,dim]
            if x_est<center_center+radius and \
                x_est>center_center-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    data_file_path = os.path.join(script_dir, '../../data_eval3.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    trace_list = []
    E_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        init = data[i][2]
        for tmp in init:
            E_list.append(tmp[1])
    # Getting Model for center center model
    state_list = np.array(state_list)
    E_list = np.array(E_list)

    
    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i, dim]
        center_center = cc[0]*x + cc[1]*y + cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        traces = trace_list[i]
        print(center_center, radius)
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,dim]
            if x_est<center_center+radius and \
                x_est>center_center-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    plt.show()
