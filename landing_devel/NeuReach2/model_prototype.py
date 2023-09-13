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
trace_mean_list = []
r_list = []
r_x_list = []
r_y_list = []
r_z_list = []
r_roll_list = []
r_pitch_list = []
r_yaw_list = []
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
    max_r_y = 0
    max_r_z = 0
    max_r_roll = 0
    max_r_pitch = 0
    max_r_yaw = 0
    trace_mean = np.mean(traces, axis=0)
    trace_mean_list.append(trace_mean)
    for j in range(traces.shape[0]):
        r = np.linalg.norm(traces[j,:]-trace_mean)
        trace_diff = np.abs(traces[j,:]-trace_mean)
        max_r = max(r, max_r)
        max_r_x = max(trace_diff[0], max_r_x)
        max_r_y = max(trace_diff[1], max_r_y)
        max_r_z = max(trace_diff[2], max_r_z)
        max_r_roll = max(trace_diff[3], max_r_roll)
        max_r_pitch = max(trace_diff[4], max_r_pitch)
        max_r_yaw = max(trace_diff[5], max_r_yaw)
    r_list.append(max_r)
    r_x_list.append(max_r_x)
    r_y_list.append(max_r_y)
    r_z_list.append(max_r_z)
    r_roll_list.append(max_r_roll)
    r_pitch_list.append(max_r_pitch)
    r_yaw_list.append(max_r_yaw)

bin = []
bin_lb_list, bin_step = np.linspace(0, 0.35, 51, retstep=True)
Er_list = np.array(Er_list)
r_list = np.array(r_list)
r_x_list = np.array(r_x_list)
r_y_list = np.array(r_y_list)
r_z_list = np.array(r_z_list)
r_roll_list = np.array(r_roll_list)
r_pitch_list = np.array(r_pitch_list)
r_yaw_list = np.array(r_yaw_list)
Er_center_list = []
bound_list = []
bound_x_list = []
bound_y_list = []
bound_z_list = []
bound_roll_list = []
bound_pitch_list = []
bound_yaw_list = []
for i in range(len(bin_lb_list)-1):
    bin_lb = bin_lb_list[i]
    bin_ub = bin_lb + bin_step 
    idx = np.where((bin_lb<Er_list) & (Er_list<bin_ub))[0]
    Er_center_list.append((bin_lb+bin_ub)/2)
    r_sub_list = r_list[idx]
    percentile = np.percentile(r_sub_list, 80)
    bound_list.append(percentile)

    r_x_sub_list = r_x_list[idx]
    percentile_x = np.percentile(r_x_sub_list, 80)
    bound_x_list.append(percentile_x)
    r_y_sub_list = r_y_list[idx]
    percentile_y = np.percentile(r_y_sub_list, 80)
    bound_y_list.append(percentile_y)
    r_z_sub_list = r_z_list[idx]
    percentile_z = np.percentile(r_z_sub_list, 80)
    bound_z_list.append(percentile_z)
    r_roll_sub_list = r_roll_list[idx]
    percentile_roll = np.percentile(r_roll_sub_list, 80)
    bound_roll_list.append(percentile_roll)
    r_pitch_sub_list = r_pitch_list[idx]
    percentile_pitch = np.percentile(r_pitch_sub_list, 80)
    bound_pitch_list.append(percentile_pitch)
    r_yaw_sub_list = r_yaw_list[idx]
    percentile_yaw = np.percentile(r_yaw_sub_list, 80)
    bound_yaw_list.append(percentile_yaw)


xdata = Er_center_list[:-1] 
ydata = bound_list[:-1]
popt1, _ = curve_fit(func_x1, xdata, ydata)
# popt2, _ = curve_fit(func_x2, xdata, ydata)

ydata = bound_x_list[:-1]
popt1_x, _ = curve_fit(func_x1, xdata, ydata)
# popt2_x, _ = curve_fit(func_x2, xdata, ydata)
plt.figure(9)
plt.plot(Er_list, r_x_list, 'b*')
plt.plot(Er_list, popt1_x[0]*np.sqrt(Er_list), 'r*')

plt.figure(10)
plt.plot(xdata, ydata, 'b*')
plt.plot(xdata, popt1_x[0]*np.sqrt(xdata), 'r*')

Er_center_list = np.array(Er_center_list)

plt.figure(0)
plt.plot(Er_center_list, bound_list, '*')
plt.plot(Er_center_list, popt1[0]*np.sqrt(Er_center_list),'b')

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
        percentile = np.percentile(tmp, 90)
        state_list_process = np.concatenate((state_list_process,X_partition[tmp<percentile]))
        Ec_list_process = np.concatenate((Ec_list_process,E_partition[tmp<percentile]))
        trace_mean_list_process = np.concatenate((trace_mean_list_process,trace_mean_partition[tmp<percentile]))

X_process = np.vstack((state_list_process, Ec_list_process)).T
Y_process = trace_mean_list_process

fig6 = plt.figure(6)
plt.plot(state_list_process, trace_mean_list_process, 'b.')

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(projection='3d')
ax5.scatter(state_list_process, Ec_list_process, trace_mean_list_process, 'b')

model_center_center = LinearRegression()
model_center_center.fit(X_process,Y_process)
# predictions = model_center_center.predict(X)

model_error = np.abs(trace_mean_list_process - model_center_center.predict(X_process))
X_center_radius = np.vstack((state_list_process, Ec_list_process, state_list_process*Ec_list_process, state_list_process**2, Ec_list_process**2), dtype=np.float32).T
Y_center_radius = model_error

X_center_radius = sm.add_constant(X_center_radius)
model_center_radius = sm.QuantReg(Y_center_radius, X_center_radius) 
result = model_center_radius.fit(q=0.95)
coefficient_center_radius = result.params 
print("Coefficients:", coefficient_center_radius)

# model_center_radius = QuantileRegressor(quantile=0.99)
# model_center_radius.fit(X_radius, Y_radius)

fig7 = plt.figure(7)
ax7 = fig7.add_subplot(projection='3d')
ax7.scatter(state_list_process, Ec_list_process, model_error, 'b')
predict_r = X_center_radius[:,0]*coefficient_center_radius[0] \
    + X_center_radius[:,1]*coefficient_center_radius[1] \
    + X_center_radius[:,2]*coefficient_center_radius[2] \
    + X_center_radius[:,3]*coefficient_center_radius[3] \
    + X_center_radius[:,4]*coefficient_center_radius[4] \
    + X_center_radius[:,5]*coefficient_center_radius[5]
ax7.scatter(state_list_process, Ec_list_process, predict_r, 'r')

# The simplest R is X -> 2^Y. In this case, X is the ground truth state, 2^Y is the radius from the model center center
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
quantile = 0.95
model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
result = model_radius.fit(q=quantile)
coefficient_radius = result.params

plt.figure(8)
plt.plot(X_radius, Y_radius, 'b*')
plt.plot(X_radius, coefficient_radius[0] + X_radius*coefficient_radius[1], 'r')

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(state_list[:,0], Ec_list, trace_mean_list[:,0], 'b')
ax1.scatter(state_list[:,0], Ec_list, model_center_center.predict(X), 'r')

fig2 = plt.figure(2)
error_list = np.abs(model_center_center.predict(X) - trace_mean_list[:,0])
plt.plot(Ec_list, error_list, 'b*')

fig3 = plt.figure(3)
error_list = np.abs(model_center_center.predict(X) - trace_mean_list[:,0])
plt.plot(state_list[:,0], error_list, 'b*')

fig4 = plt.figure(4)
plt.plot(state_list[:,0], trace_mean_list[:,0], 'b.')
plt.plot(state_list[:,0], model_center_center.predict(X), 'r.')

# The center of perception contract. 
model_center_center # Input to this function is the ground truth state and center of range of environmental parameter

# The radius of possible center 
model_center_radius 
coefficient_center_radius # Input to this function is the ground truth state and center of range of environmental parameter

# The radius of perception contract 
model_radius 
coefficient_radius # Input to this function is the ground truth state
model_radius_decay = lambda r: 1/(np.sqrt(0.35))*np.sqrt(r) # Input to this function is the radius of environmental parameters

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

data_file_path = os.path.join(script_dir, 'data.pickle')
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

data_file_path = os.path.join(script_dir, 'data_eval.pickle')
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
    print(center_radius, radius, center_radius+radius)
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
