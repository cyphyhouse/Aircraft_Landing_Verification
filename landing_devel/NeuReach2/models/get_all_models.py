import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

from model_x import compute_model_x
from model_y import compute_model_y
from model_z import compute_model_z
from model_yaw import compute_model_yaw
from model_pitch import compute_model_pitch


def apply_model(model, point, Ec, Er):
    dim = model['dim']
    coef_cc = model['coef_center_center']
    coef_cr = model['coef_center_radius']
    coef_r = model['coef_radius']

    if dim == 'x':
        point = point[0]
    elif dim == 'y':
        point = point[(0,1),]
    elif dim == 'z':
        point = point[(0,2),]
    elif dim == 'yaw':
        point = point[(0,5),]
    elif dim == 'pitch':
        point = point[(0,4),]

    model_radius_decay = lambda r: (1/np.sqrt(0.35))*np.sqrt(r)
    if dim == 'x':
        x = point
        ec = Ec[0]
        er = Er[0]
        center_center = coef_cc[0] * x + coef_cc[1] * ec + coef_cc[2]
        center_radius = coef_cr[0] \
            + x*coef_cr[1] \
            + ec*coef_cr[2] \
            + x*ec*coef_cr[3] \
            + x**2*coef_cr[4]\
            + ec**2*coef_cr[4]
        radius = (coef_r[0]+coef_r[1]*x) * model_radius_decay(er)
        return center_center, center_radius + radius 
    else:
        x = point[0]
        y = point[1]
        ec = Ec[0]
        er = Er[0]
        center_center = coef_cc[0]*x+coef_cc[1]*y+coef_cc[2]*ec+coef_cc[3]
        center_radius = coef_cr[0] \
            + x*coef_cr[1] \
            + y*coef_cr[2] \
            + ec*coef_cr[3] \
            + x*ec*coef_cr[4] \
            + y*ec*coef_cr[5] \
            + x*y*coef_cr[6] \
            + x**2*coef_cr[7]\
            + y**2*coef_cr[8] \
            + ec**2*coef_cr[9]
        radius = (coef_r[0] + coef_r[1]*x + coef_r[2]*y)*model_radius_decay(er)
        return center_center, center_radius+radius

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_train.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    
    mcc, ccr, cr = compute_model_x(data, 0.9, 0.95, 0.97)
    model_x = {
        'dim': 'x',
        'coef_center_center':mcc.coef_.tolist()+[mcc.intercept_],
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_x.json'),'w+') as f:
        json.dump(model_x, f, indent=4)

    mcc, ccr, cr = compute_model_y(data, 0.9, 0.95, 0.97)
    model_y = {
        'dim': 'y',
        'coef_center_center':mcc.coef_.tolist()+[mcc.intercept_],
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_y.json'),'w+') as f:
        json.dump(model_y, f, indent=4)

    mcc, ccr, cr = compute_model_z(data, 0.9, 0.95, 0.98)
    model_z = {
        'dim': 'z',
        'coef_center_center':mcc.coef_.tolist()+[mcc.intercept_],
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_z.json'),'w+') as f:
        json.dump(model_z, f, indent=4)

    mcc, ccr, cr = compute_model_yaw(data, 0.7, 0.95, 0.97)
    model_yaw = {
        'dim': 'yaw',
        'coef_center_center':mcc.coef_.tolist()+[mcc.intercept_],
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_yaw.json'),'w+') as f:
        json.dump(model_yaw, f, indent=4)

    mcc, ccr, cr = compute_model_pitch(data, 0.9, 0.95, 0.985)
    model_pitch = {
        'dim': 'pitch',
        'coef_center_center':mcc.coef_.tolist()+[mcc.intercept_],
        'coef_center_radius':ccr.tolist(),
        'coef_radius': cr.tolist()
    }
    with open(os.path.join(script_dir,'model_pitch.json'),'w+') as f:
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
            traces[j,5] > c_yaw-r_yaw and traces[j,5] < c_yaw+r_yaw and \
            traces[j,4] > c_pitch-r_pitch and traces[j,4] < c_pitch+r_pitch:
                contained_data += 1  
            if traces[j,0] > c_x-r_x and traces[j,0] < c_x+r_x:
                contained_data_x += 1
            if traces[j,1] > c_y-r_y and traces[j,1] < c_y+r_y:
                contained_data_y += 1
            if traces[j,2] > c_z-r_z and traces[j,2] < c_z+r_z:
                contained_data_z += 1
            if traces[j,5] > c_yaw-r_yaw and traces[j,5] < c_yaw+r_yaw:
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

