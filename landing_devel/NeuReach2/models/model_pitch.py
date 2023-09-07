import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

def compute_model_z(data):
    model_dim = 4
    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)

    # Getting Model for center center model
    Ec_list = np.array(Ec_list)
    state_list = np.array(state_list)
    state_list = state_list[:,(0,model_dim)]
    trace_mean_list = np.array(trace_mean_list)
    trace_mean_list = trace_mean_list[:, (0,model_dim)]

    X = np.vstack((state_list[:,0], state_list[:,1], Ec_list)).T
    Y = trace_mean_list[:,1]

    state_list_process = np.zeros((2,0))
    Ec_list_process = np.array([])
    trace_mean_list_process = np.array([])
    Xp0 = np.arange(-3000, 2000, 1000)
    Xp1 = np.arange(-9, 3, 1)
    Ep = np.arange(0.5, 1.2, 0.1)
    for i in range(Xp0.shape[0]):
        for j in range(Xp1.shape[0]):
            for k in range(Ep.shape[0]):
                idx = np.where((state_list[:,0]>Xp0[i]) &\
                                (state_list[:,0]<Xp0[i]+1000) &\
                                (state_list[:,1]>Xp1[j]) &\
                                (state_list[:,1]<Xp1[j]+10) &\
                                (Ec_list>Ep[k]) &\
                                (Ec_list<Ep[k]+0.1))[0]
                X_partition = state_list[idx, :]
                dim_partition = state_list[idx, 1]
                E_partition = Ec_list[idx]
                trace_mean_partition = trace_mean_list[idx,1]
                tmp = np.abs(trace_mean_partition - dim_partition)
                if tmp.size<=0:
                    continue
                percentile = np.percentile(tmp, 90)
                state_list_process = np.hstack((state_list_process,X_partition[tmp<percentile,:].T))
                Ec_list_process = np.concatenate((Ec_list_process,E_partition[tmp<percentile]))
                trace_mean_list_process = np.concatenate((trace_mean_list_process,trace_mean_partition[tmp<percentile]))

    X_process = np.vstack((state_list_process, Ec_list_process)).T # X, Z, Ec
    Y_process = trace_mean_list_process

    model_center_center = LinearRegression()
    model_center_center.fit(X_process,Y_process)
    # -------------------------------------

    # Getting Model for Center Radius
    model_error = np.abs(trace_mean_list_process - model_center_center.predict(X_process))
    X_center_radius = np.vstack(
        (
            state_list_process[0,:],                            # x 
            state_list_process[1,:],                            # z
            Ec_list_process,                                    # ec
            state_list_process[0,:]*Ec_list_process,            # x*ec
            state_list_process[1,:]*Ec_list_process,            # z*ec
            state_list_process[0,:]*state_list_process[1,:],    # x*z
            state_list_process[0,:]**2,                         # x**2
            state_list_process[1,:]**2,                         # z**2
            Ec_list_process**2                                  # ec**2
        ), dtype=np.float32).T
    Y_center_radius = model_error

    X_center_radius = sm.add_constant(X_center_radius)
    model_center_radius = sm.QuantReg(Y_center_radius, X_center_radius) 
    result = model_center_radius.fit(q=0.95, max_iter=5000)
    coefficient_center_radius = result.params 
    print("Coefficients:", coefficient_center_radius)
    # -------------------------------------

    # Getting Model for Radius
    x_state_value = state_list[:,(0,1)]
    center_center = model_center_center.predict(X)
    trace_list_combine = trace_list[0][:,(0,model_dim)]
    X_radius = [x_state_value[0,:]]*trace_list[0].shape[0]
    mean_radius = [trace_mean_list[0,:]]*trace_list[0].shape[0]
    tmp = [center_center[0]]*trace_list[0].shape[0]
    for i in range(1, len(trace_list)):
        trace_list_combine = np.vstack((trace_list_combine, trace_list[i][:,(0,model_dim)]))
        X_radius += ([x_state_value[i,:]]*trace_list[i].shape[0])
        mean_radius += ([trace_mean_list[i,:]]*trace_list[i].shape[0])
        tmp += ([center_center[i]]*trace_list[i].shape[0])
    X_radius = np.array(X_radius)
    mean_radius = np.array(mean_radius)
    tmp = np.array(tmp)
    Y_radius = np.abs(trace_list_combine[:,1]-mean_radius[:,1])
    # Y_radius = np.abs(trace_list_combine[:,1]-X_radius[:,1])
    quantile = 0.985
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
    result = model_radius.fit(q=quantile, max_iter=5000)
    coefficient_radius = result.params
    # -------------------------------------
    return model_center_center, coefficient_center_radius, coefficient_radius

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
    state_list = np.array(state_list)[:,(0,4)]

    model_center_center, coefficient_center_radius, coefficient_radius = compute_model_z(data)

    res = {
        'dim': 'pitch',
        'coef_center_center':model_center_center.coef_.tolist()+[model_center_center.intercept_],
        'coef_center_radius':coefficient_center_radius.tolist(),
        'coef_radius': coefficient_radius.tolist()
    }
    with open(os.path.join(script_dir,'model_pitch.json'),'w+') as f:
        json.dump(res, f, indent=4)

    # Testing the obtained models
    # The center of perception contract. 
    model_center_center # Input to this function is the ground truth state and center of range of environmental parameter

    # The radius of possible center 
    # model_center_radius 
    coefficient_center_radius # Input to this function is the ground truth state and center of range of environmental parameter

    # The radius of perception contract 
    # model_radius 
    coefficient_radius # Input to this function is the ground truth state
    model_radius_decay = lambda r: (1/np.sqrt(0.35))*np.sqrt(r) # Input to this function is the radius of environmental parameters

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i,1]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, y, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + y*coefficient_center_radius[2] \
            + ec*coefficient_center_radius[3] \
            + x*ec*coefficient_center_radius[4] \
            + y*ec*coefficient_center_radius[5] \
            + x*y*coefficient_center_radius[6] \
            + x**2*coefficient_center_radius[7]\
            + y**2*coefficient_center_radius[8] \
            + ec**2*coefficient_center_radius[9]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x + coefficient_radius[2]*y)*model_radius_decay(er)
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,4]
            if x_est<center_center+center_radius+radius and \
                x_est>center_center-center_radius-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    data_file_path = os.path.join(script_dir, '../data.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
    state_list = np.array(state_list)[:,(0,4)]

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i,1]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, y, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + y*coefficient_center_radius[2] \
            + ec*coefficient_center_radius[3] \
            + x*ec*coefficient_center_radius[4] \
            + y*ec*coefficient_center_radius[5] \
            + x*y*coefficient_center_radius[6] \
            + x**2*coefficient_center_radius[7]\
            + y**2*coefficient_center_radius[8] \
            + ec**2*coefficient_center_radius[9]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x + coefficient_radius[2]*y)*model_radius_decay(er)
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,4]
            if x_est<center_center+center_radius+radius and \
                x_est>center_center-center_radius-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    data_file_path = os.path.join(script_dir, '../data_eval.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
    state_list = np.array(state_list)[:,(0,4)]

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i,1]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, y, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + y*coefficient_center_radius[2] \
            + ec*coefficient_center_radius[3] \
            + x*ec*coefficient_center_radius[4] \
            + y*ec*coefficient_center_radius[5] \
            + x*y*coefficient_center_radius[6] \
            + x**2*coefficient_center_radius[7]\
            + y**2*coefficient_center_radius[8] \
            + ec**2*coefficient_center_radius[9]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x + coefficient_radius[2]*y)*model_radius_decay(er)
        traces = trace_list[i]
        # print(center_radius, radius, center_radius+radius)
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,4]
            if x_est<center_center+center_radius+radius and \
                x_est>center_center-center_radius-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    # plt.show()
