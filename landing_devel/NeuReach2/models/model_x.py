import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

def compute_model_x(data, pcc=0.9, pcr=0.95, pr=0.95):
    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    r_list = []
    r_x_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        max_r = 0
        max_r_x = 0
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)
        for j in range(traces.shape[0]):
            r = np.linalg.norm(traces[j,:]-trace_mean)
            trace_diff = np.abs(traces[j,:]-trace_mean)
            max_r = max(r, max_r)
            max_r_x = max(trace_diff[0], max_r_x)
        r_list.append(max_r)
        r_x_list.append(max_r_x)

    # Getting Model for center center model
    Ec_list = np.array(Ec_list)
    state_list = np.array(state_list)
    trace_mean_list = np.array(trace_mean_list)

    X = np.vstack((state_list[:,0], Ec_list)).T
    Y = trace_mean_list[:,0]

    state_list_process = np.array([])
    Ec_list_process = np.array([])
    trace_mean_list_process = np.array([])
    Xp = np.arange(-3000, -2000, 100)
    Ep = np.arange(0.5, 1.2, 0.1)
    for i in range(Xp.shape[0]):
        for j in range(Ep.shape[0]):
            idx = np.where((state_list[:,0]>Xp[i]) &\
                            (state_list[:,0]<Xp[i]+1000) &\
                            (Ec_list>Ep[j]) &\
                            (Ec_list<Ep[j]+0.1))[0]
            X_partition = state_list[idx, 0]
            E_partition = Ec_list[idx]
            trace_mean_partition = trace_mean_list[idx,0]
            tmp = np.abs(trace_mean_partition - X_partition)
            percentile = np.percentile(tmp, pcc*100)
            state_list_process = np.concatenate((state_list_process,X_partition[tmp<percentile]))
            Ec_list_process = np.concatenate((Ec_list_process,E_partition[tmp<percentile]))
            trace_mean_list_process = np.concatenate((trace_mean_list_process,trace_mean_partition[tmp<percentile]))

    X_process = np.vstack((state_list_process, Ec_list_process)).T
    Y_process = trace_mean_list_process

    # fig6 = plt.figure(6)
    # plt.plot(state_list_process, trace_mean_list_process, 'b.')

    # fig5 = plt.figure(5)
    # ax5 = fig5.add_subplot(projection='3d')
    # ax5.scatter(state_list_process, Ec_list_process, trace_mean_list_process, 'b')

    model_center_center = LinearRegression()
    model_center_center.fit(X_process,Y_process)
    # -------------------------------------

    # Getting Model for Center Radius
    model_error = np.abs(trace_mean_list_process - model_center_center.predict(X_process))
    X_center_radius = np.vstack((state_list_process, Ec_list_process, state_list_process*Ec_list_process, state_list_process**2, Ec_list_process**2), dtype=np.float32).T
    Y_center_radius = model_error

    X_center_radius = sm.add_constant(X_center_radius)
    model_center_radius = sm.QuantReg(Y_center_radius, X_center_radius) 
    result = model_center_radius.fit(q=pcr)
    coefficient_center_radius = result.params 
    print("Coefficients:", coefficient_center_radius)
    # -------------------------------------

    # Getting Model for Radius
    x_state_value = state_list[:,0]
    center_center = model_center_center.predict(X)
    trace_list_radius = trace_list[0]
    X_radius = [x_state_value[0]]*trace_list[0].shape[0]
    mean_radius = [trace_mean_list[0]]*trace_list[0].shape[0]
    tmp = [center_center[0]]*trace_list[0].shape[0]
    for i in range(1, len(trace_list)):
        trace_list_radius = np.vstack((trace_list_radius, trace_list[i]))
        X_radius += ([x_state_value[i]]*trace_list[i].shape[0])
        mean_radius += ([trace_mean_list[i]]*trace_list[i].shape[0])
        tmp += ([center_center[i]]*trace_list[i].shape[0])
    X_radius = np.array(X_radius)
    mean_radius = np.array(mean_radius)
    tmp = np.array(tmp)
    Y_radius = np.abs(trace_list_radius[:,0]-mean_radius[:,0])
    # Y_radius = np.abs(trace_list_radius[:,0]-X_radius)
    quantile = pr
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
    result = model_radius.fit(q=quantile)
    coefficient_radius = result.params
    return model_center_center, coefficient_center_radius, coefficient_radius
# -------------------------------------

# Testing the obtained models
# The center of perception contract. 
# model_center_center # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of possible center 
# model_center_radius 
# coefficient_center_radius # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of perception contract 
# model_radius 
# coefficient_radius # Input to this function is the ground truth state

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_train.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    r_list = []
    r_x_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec[0]) 
        Er_list.append(Er[0])
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        max_r = 0
        max_r_x = 0
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)
        for j in range(traces.shape[0]):
            r = np.linalg.norm(traces[j,:]-trace_mean)
            trace_diff = np.abs(traces[j,:]-trace_mean)
            max_r = max(r, max_r)
            max_r_x = max(trace_diff[0], max_r_x)
        r_list.append(max_r)
        r_x_list.append(max_r_x)
    state_list = np.array(state_list)
    Er_list = np.array(Er_list)
    Ec_list=  np.array(Ec_list)

    model_center_center, coefficient_center_radius, coefficient_radius = compute_model_x(data)

    res = {
        'dim': 'x',
        'coef_center_center':model_center_center.coef_.tolist()+[model_center_center.intercept_],
        'coef_center_radius':coefficient_center_radius.tolist(),
        'coef_radius': coefficient_radius.tolist()
    }
    with open(os.path.join(script_dir,'model_x.json'),'w+') as f:
        json.dump(res, f, indent=4)

    model_radius_decay = lambda r: (1/np.sqrt(0.35))*np.sqrt(r) # Input to this function is the radius of environmental parameters

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + ec*coefficient_center_radius[2] \
            + x*ec*coefficient_center_radius[3] \
            + x**2*coefficient_center_radius[4]\
            + ec**2*coefficient_center_radius[4]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x)*model_radius_decay(er)
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,0]
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
    state_list = np.array(state_list)
    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + ec*coefficient_center_radius[2] \
            + x*ec*coefficient_center_radius[3] \
            + x**2*coefficient_center_radius[4]\
            + ec**2*coefficient_center_radius[4]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x)*model_radius_decay(er)
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,0]
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
    state_list = np.array(state_list)
    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = model_center_center.predict(np.array([x, ec]).reshape(1,-1))
        center_radius = coefficient_center_radius[0] \
            + x*coefficient_center_radius[1] \
            + ec*coefficient_center_radius[2] \
            + x*ec*coefficient_center_radius[3] \
            + x**2*coefficient_center_radius[4]\
            + ec**2*coefficient_center_radius[4]
        radius = (coefficient_radius[0] + coefficient_radius[1]*x)*model_radius_decay(er)
        traces = trace_list[i]
        # print(center_radius, radius, center_radius+radius)
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,0]
            if x_est<center_center+center_radius+radius and \
                x_est>center_center-center_radius-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)

    plt.show()
