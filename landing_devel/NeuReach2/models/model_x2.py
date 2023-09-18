import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

model_radius_decay = lambda r, r_max: (1/np.sqrt(r_max))*np.sqrt(r) # 0.25Input to this function is the radius of environmental parameters

def compute_model_x(data, pcc=0.9, pcr=0.95, pr=0.95):
    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec) 
        Er_list.append(Er)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)

    # Getting Model for center center model
    Ec_list = np.array(Ec_list)
    state_list = np.array(state_list)
    trace_mean_list = np.array(trace_mean_list)

    X = np.hstack((state_list[:,0:1], Ec_list))
    Y = trace_mean_list[:,0]

    state_list_process = np.array([])
    Ec_list_process = np.zeros((0,2))
    trace_mean_list_process = np.array([])
    Xp = np.arange(-3000, -2000, 100)
    Ep1 = np.arange(0.5, 1.2, 0.1)
    Ep2 = np.arange(0,0.5,0.1)
    total_num = 0
    for i in range(Xp.shape[0]):
        for j in range(Ep1.shape[0]):
            for k in range(Ep2.shape[0]):
                idx = np.where((state_list[:,0]>Xp[i]) &\
                                (state_list[:,0]<(Xp[i]+100)) &\
                                (Ec_list[:,0]>Ep1[j]) &\
                                (Ec_list[:,0]<(Ep1[j]+0.1)) &\
                                (Ec_list[:,1]>Ep2[k]) &\
                                (Ec_list[:,1]<(Ep2[k]+0.1)))[0]
                X_partition = state_list[idx, 0]
                E_partition = Ec_list[idx,:]
                total_num += X_partition.size
                trace_mean_partition = trace_mean_list[idx,0]
                tmp = np.abs(trace_mean_partition - X_partition)
                if tmp.size!=0:
                    percentile = np.percentile(tmp, pcc*100)
                    state_list_process = np.concatenate((state_list_process,X_partition[tmp<percentile]))
                    Ec_list_process = np.vstack((Ec_list_process,E_partition[tmp<percentile]))
                    trace_mean_list_process = np.concatenate((trace_mean_list_process,trace_mean_partition[tmp<percentile]))
    print(total_num)
    X_process = np.hstack((state_list_process.reshape((-1,1)), Ec_list_process))
    Y_process = trace_mean_list_process

    # fig6 = plt.figure(6)
    # plt.plot(state_list_process, trace_mean_list_process, 'b.')

    # fig5 = plt.figure(5)
    # ax5 = fig5.add_subplot(projection='3d')
    # ax5.scatter(state_list_process, Ec_list_process, trace_mean_list_process, 'b')

    mcc = LinearRegression()
    mcc.fit(X_process,Y_process)
    # -------------------------------------

    # Getting Model for Center Radius
    model_error = np.abs(trace_mean_list_process - mcc.predict(X_process))
    X_center_radius = np.hstack((
        state_list_process.reshape((-1,1)), 
        Ec_list_process[:,0].reshape((-1,1)), 
        Ec_list_process[:,1].reshape((-1,1)), 
        (state_list_process*Ec_list_process[:,0]).reshape((-1,1)), 
        (state_list_process*Ec_list_process[:,1]).reshape((-1,1)), 
        (Ec_list_process[:,0]*Ec_list_process[:,1]).reshape((-1,1)), 
        (state_list_process**2).reshape((-1,1)), 
        (Ec_list_process[:,0]**2).reshape((-1,1)),
        (Ec_list_process[:,1]**2).reshape((-1,1)),
    ), dtype=np.float32)
    Y_center_radius = model_error

    X_center_radius = sm.add_constant(X_center_radius)
    model_center_radius = sm.QuantReg(Y_center_radius, X_center_radius) 
    result = model_center_radius.fit(q=pcr)
    ccr = result.params 
    print("Coefficients:", ccr)
    # -------------------------------------

    # Getting Model for Radius
    x_state_value = state_list[:,0]
    center_center = mcc.predict(X)
    trace_list_radius = trace_list[0]
    n = trace_list[0].shape[0]
    X_radius = [x_state_value[0]]*n
    X_radius = np.zeros(len(trace_list)*n)
    mean_radius = [trace_mean_list[0]]*n
    mean_radius = np.zeros((len(trace_list)*n,trace_mean_list[0].shape[0]))
    tmp = [center_center[0]]*n
    tmp = np.zeros(len(trace_list)*n)
    trace_list_radius = np.zeros((n*len(trace_list), trace_list[0].shape[1]))
    for i in range(0, len(trace_list)):
        # trace_list_radius = np.vstack((trace_list_radius, trace_list[i]))
        # X_radius += ([x_state_value[i]]*trace_list[i].shape[0])
        # mean_radius += ([trace_mean_list[i]]*trace_list[i].shape[0])
        # tmp += ([center_center[i]]*trace_list[i].shape[0])
        trace_list_radius[i*n:(i+1)*n,:] = trace_list[i]
        X_radius[i*n:(i+1)*n] = x_state_value[i]
        mean_radius[i*n:(i+1)*n,:] = trace_mean_list[i]
        tmp[i*n:(i+1)*n] = center_center[i]
    X_radius = np.array(X_radius)
    mean_radius = np.array(mean_radius)
    tmp = np.array(tmp)
    Y_radius = np.abs(trace_list_radius[:,0]-mean_radius[:,0])
    # Y_radius = np.abs(trace_list_radius[:,0]-X_radius)
    quantile = pr
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
    result = model_radius.fit(q=quantile)
    cr = result.params

    ccc = mcc.coef_.tolist()+[mcc.intercept_]
    min_cc = 0
    min_cr = 0
    min_r = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = ccc[0]*x + ccc[1]*ec[0] + ccc[2]*ec[1] + ccc[3]
        if center_center < min_cc:
            min_cc = center_center
        center_radius = ccr[0] \
            + x*ccr[1] \
            + ec[0]*ccr[2] \
            + ec[1]*ccr[3] \
            + x*ec[0]*ccr[4] \
            + x*ec[1]*ccr[5] \
            + ec[0]*ec[1]*ccr[6] \
            + x**2*ccr[7]\
            + ec[0]**2*ccr[8]\
            + ec[1]**2*ccr[9]
        if center_radius< min_cr:
            min_cr = center_radius
        radius = (cr[0] + cr[1]*x)*model_radius_decay(er[0], 0.35)*model_radius_decay(er[1], 0.25)
        if radius < min_r:
            min_r = radius
    ccr[0] += (-min_cr)
    cr[0] += (-min_r)
    return ccc, ccr, cr
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
    data_file_path = os.path.join(script_dir, '../data_train2.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec) 
        Er_list.append(Er)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)
    # Getting Model for center center model
    Ec_list = np.array(Ec_list)
    state_list = np.array(state_list)
    trace_mean_list = np.array(trace_mean_list)

    ccc, ccr, cr = compute_model_x(data, pcc = 0.4, pcr=0.8, pr=0.9)
    res = {
        'dim': 'x',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_x2.json'),'w+') as f:
        json.dump(res, f, indent=4)

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = ccc[0]*x + ccc[1]*ec[0] + ccc[2]*ec[1] + ccc[3]
        center_radius = ccr[0] \
            + x*ccr[1] \
            + ec[0]*ccr[2] \
            + ec[1]*ccr[3] \
            + x*ec[0]*ccr[4] \
            + x*ec[1]*ccr[5] \
            + ec[0]*ec[1]*ccr[6] \
            + x**2*ccr[7]\
            + ec[0]**2*ccr[8]\
            + ec[1]**2*ccr[9]
        radius = (cr[0] + cr[1]*x)*model_radius_decay(er[0], 0.35)*model_radius_decay(er[1], 0.25)
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

    data_file_path = os.path.join(script_dir, '../data_eval2.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_list = []
    state_list = []
    Er_list = []
    Ec_list = []
    trace_list = []
    trace_mean_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        Ec_list.append(Ec) 
        Er_list.append(Er)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        trace_mean = np.mean(traces, axis=0)
        trace_mean_list.append(trace_mean)

    # Getting Model for center center model
    Ec_list = np.array(Ec_list)
    state_list = np.array(state_list)
    trace_mean_list = np.array(trace_mean_list)
    
    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        ec = Ec_list[i]
        er = Er_list[i]
        center_center = ccc[0]*x + ccc[1]*ec[0] + ccc[2]*ec[1] + ccc[3]
        center_radius = ccr[0] \
            + x*ccr[1] \
            + ec[0]*ccr[2] \
            + ec[1]*ccr[3] \
            + x*ec[0]*ccr[4] \
            + x*ec[1]*ccr[5] \
            + ec[0]*ec[1]*ccr[6] \
            + x**2*ccr[7]\
            + ec[0]**2*ccr[8]\
            + ec[1]**2*ccr[9]
        radius = (cr[0] + cr[1]*x)*model_radius_decay(er[0], 0.35)*model_radius_decay(er[1], 0.25)
        print(center_radius, radius)
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

    plt.show()
