import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 

def func_x1(x,a):
    y = a*np.sqrt(x)
    return y 

def func_x2(x,a, b):
    y = a*(1-1/(12.5*x+1))
    return y

def func_x_c(x, Ec, a, b, c):
    y = a*x+b*Ec+c
    return y

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, 'data_train2.pickle')
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
    # for j in range(trace_list.shape[1]):
    trace_seg = trace_list[i,:,0]
    state_seg = np.zeros(trace_list.shape[0])
    state_seg[:] = state_list[i,0]
    e1_seg = e_list[i,:,0]
    e2_seg = e_list[i,:,1]
    idx = np.where(
        (e2_seg>0.2) & \
        (e2_seg<0.3) & \
        (e1_seg>0.7) & \
        (e1_seg<0.8))
    trace_seg = trace_seg[idx]
    state_seg = state_seg[idx]
    plt.plot(state_seg, trace_seg, 'b*')

plt.show()

