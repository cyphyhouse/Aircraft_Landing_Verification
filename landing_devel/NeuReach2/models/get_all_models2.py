import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

from model_x2 import compute_model_x
from model_y2 import compute_model_y
from model_z2 import compute_model_z
from model_yaw2 import compute_model_yaw
from model_pitch2 import compute_model_pitch

model_radius_decay = lambda r, r_max: (1/np.sqrt(r_max))*np.sqrt(r) # 0.25Input to this function is the radius of environmental parameters

def apply_model(model, point, Ec, Er):
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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_train2.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    
    ccc, ccr, cr = compute_model_x(data, pcc = 0.4, pcr=0.75, pr=0.75)
    model_x = {
        'dim': 'x',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_x3.json'),'w+') as f:
        json.dump(model_x, f, indent=4)

    ccc, ccr, cr = compute_model_y(data, pcc = 0.5, pcr=0.6, pr=0.78)
    model_y = {
        'dim': 'y',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_y3.json'),'w+') as f:
        json.dump(model_y, f, indent=4)

    ccc, ccr, cr = compute_model_z(data, pcc = 0.5, pcr=0.8, pr=0.8)
    model_z = {
        'dim': 'z',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_z3.json'),'w+') as f:
        json.dump(model_z, f, indent=4)

    ccc, ccr, cr = compute_model_yaw(data, pcc = 0.5, pcr=0.75, pr=0.77)
    # ccc = mcc.coef_.tolist()+[mcc.intercept_]

    model_yaw = {
        'dim': 'yaw',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_yaw3.json'),'w+') as f:
        json.dump(model_yaw, f, indent=4)

    ccc, ccr, cr = compute_model_pitch(data, pcc = 0.5, pcr=0.75, pr=0.785)
    model_pitch = {
        'dim': 'pitch',
        'coef_center_center':ccc,
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_pitch3.json'),'w+') as f:
        json.dump(model_pitch, f, indent=4)

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
    plt.figure(0)
    plt.title('y')
    plt.figure(1)
    plt.title('yaw')
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

        for j in range(traces.shape[0]):
            total_data += 1 
            if traces[j,0] > c_x-r_x and traces[j,0] < c_x+r_x and \
            traces[j,1] > c_y-r_y and traces[j,1] < c_y+r_y and \
            traces[j,2] > c_z-r_z and traces[j,2] < c_z+r_z and \
            traces[j,3] > c_yaw-r_yaw and traces[j,3] < c_yaw+r_yaw and \
            traces[j,4] > c_pitch-r_pitch and traces[j,4] < c_pitch+r_pitch:
                contained_data += 1  
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

