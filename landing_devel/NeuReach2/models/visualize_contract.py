import pickle 
import os 
import numpy as np 
import json 
import matplotlib.pyplot as plt 

script_dir = os.path.dirname(os.path.realpath(__file__))

model_x_name = './model_x3.json'
model_y_name = './model_y3.json'
model_z_name = './model_z3.json'
model_yaw_name = './model_yaw3.json'
model_pitch_name = './model_pitch3.json'

with open(os.path.join(script_dir, model_x_name), 'r') as f:
    model_x = json.load(f)
with open(os.path.join(script_dir, model_y_name), 'r') as f:
    model_y = json.load(f)
with open(os.path.join(script_dir, model_z_name), 'r') as f:
    model_z = json.load(f)
with open(os.path.join(script_dir, model_pitch_name), 'r') as f:
    model_pitch = json.load(f)
with open(os.path.join(script_dir, model_yaw_name), 'r') as f:
    model_yaw = json.load(f)

def apply_model(model, point, Ec, Er):
    model_radius_decay = lambda r, r_max: (1/np.sqrt(r_max))*np.sqrt(r) # 0.25Input to this function is the radius of environmental parameters
    dim = model['dim']
    ccc = model['coef_center_center']
    ccr = model['coef_center_radius']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[0]
    elif dim == 'y':
        point = point[(0,1),]
    elif dim == 'z':
        point = point[(0,2),]
    elif dim == 'yaw':
        point = point[(0,3),]
    elif dim == 'pitch':
        point = point[(0,4),]

    if dim == 'x':
        x = point
        ec = Ec
        er = Er
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
        
        return center_center, center_radius + radius 
    else:
        x = point[0]
        y = point[1]
        ec = Ec
        er = Er
        center_center = ccc[0]*x + ccc[1]*y + ccc[2]*ec[0] + ccc[3]*ec[1] + ccc[4]
        center_radius = ccr[0] \
            + x*ccr[1] \
            + y*ccr[2] \
            + ec[0]*ccr[3] \
            + ec[1]*ccr[4] \
            + x*ec[0]*ccr[5] \
            + y*ec[0]*ccr[6] \
            + x*ec[1]*ccr[7] \
            + y*ec[1]*ccr[8] \
            + x*y*ccr[9] \
            + ec[0]*ec[1]*ccr[10] \
            + x**2*ccr[11] \
            + y**2*ccr[12] \
            + ec[0]**2*ccr[13] \
            + ec[1]**2*ccr[14]
        radius = (cr[0] + cr[1]*x + cr[2]*y)*model_radius_decay(er[0], 0.35)*model_radius_decay(er[1], 0.25)
        return center_center, center_radius+radius


data_file_path = os.path.join(script_dir, '../data_train2.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)

state_list = []
Er_list = []
Ec_list = []
est_list = []
total_data = 0
contained_data = 0
contained_data_x = 0
contained_data_y = 0
contained_data_z = 0
contained_data_yaw = 0
contained_data_pitch = 0
heat_map_correct = np.zeros((7,5))
heat_map_total = np.zeros((7,5))
for i in range(len(data)):
    X0 = data[i][0]
    x, Ec, Er = X0 
    state_list.append(x)
    Ec_list.append(Ec[0])
    Er_list.append(Er[0])
    traces = data[i][1]
    traces = np.reshape(traces, (-1,6))
    est_list.append(traces)
    x = np.array(x)
    c_x, r_x = apply_model(model_x, x, Ec, Er)
    c_y, r_y = apply_model(model_y, x, Ec, Er)
    c_z, r_z = apply_model(model_z, x, Ec, Er)
    c_yaw, r_yaw = apply_model(model_yaw, x, Ec, Er)
    c_pitch, r_pitch = apply_model(model_pitch, x, Ec, Er)

    Es = data[i][2]

    for j in range(traces.shape[0]):
        x0, E = Es[j]
        total_data += 1
        E1_idx = int(E[0]//0.1)-5
        E2_idx = int(E[1]//0.1) 
        if E1_idx == 4 and E2_idx == 0:
            print("stop")
        if traces[j,0] > c_x-r_x and traces[j,0] < c_x+r_x and \
           traces[j,1] > c_y-r_y and traces[j,1] < c_y+r_y and \
           traces[j,2] > c_z-r_z and traces[j,2] < c_z+r_z and \
           traces[j,3] > c_yaw-r_yaw and traces[j,3] < c_yaw+r_yaw and \
           traces[j,4] > c_pitch-r_pitch and traces[j,4] < c_pitch+r_pitch:
            contained_data += 1
            heat_map_correct[E1_idx, E2_idx] += 1
            heat_map_total[E1_idx, E2_idx] += 1 
        else:
            heat_map_total[E1_idx, E2_idx] += 1 

        if traces[j,0] > c_x-r_x and traces[j,0] < c_x+r_x:
            contained_data_x += 1
        if traces[j,1] > c_y-r_y and traces[j,1] < c_y+r_y:
            contained_data_y += 1
        if traces[j,2] > c_z-r_z and traces[j,2] < c_z+r_z:
            contained_data_z += 1
        if traces[j,3] > c_yaw-r_yaw and traces[j,3] < c_yaw+r_yaw:
            contained_data_yaw += 1
        if traces[j,4] > c_pitch-r_pitch and traces[j,4] < c_pitch+r_pitch:
            contained_data_pitch += 1
    # plt.figure(0)
    # plt.plot([],[],'b*')
    # plt.plot([],[],'r*')
print(contained_data/total_data)
print(contained_data_x/total_data)
print(contained_data_y/total_data)
print(contained_data_z/total_data)
print(contained_data_yaw/total_data)
print(contained_data_pitch/total_data)
heat_map_perc = heat_map_correct/heat_map_total

plt.figure(0)
plt.imshow(heat_map_perc)
plt.figure(1)
plt.imshow(heat_map_total)
plt.show()
