from models.model_x2 import compute_model_x
from models.model_y2 import compute_model_y
from models.model_z2 import compute_model_z
from models.model_yaw2 import compute_model_yaw
from models.model_pitch2 import compute_model_pitch
import os 
import pickle 
import numpy as np
import json 

def get_all_models(data):
    state_array, trace_array, E_array = data

    model_x = compute_model_x(state_array, trace_array, E_array, 0.5, 0.94)
    model_y = compute_model_y(state_array, trace_array, E_array, 0.5, 0.94)
    model_z = compute_model_z(state_array, trace_array, E_array, 0.5, 0.94)
    model_yaw = compute_model_yaw(state_array, trace_array, E_array, 0.5, 0.94)
    model_pitch = compute_model_pitch(state_array, trace_array, E_array, 0.5, 0.94)

    return model_x, model_y, model_z, model_yaw, model_pitch

def pre_process_data(data):

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
    return state_array, trace_array, E_array 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_train5.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    data = pre_process_data(data)

    model_x, model_y, model_z, model_yaw, model_pitch = get_all_models(data)
    with open(os.path.join(script_dir, './models/model_x2.json'), 'w+') as f:
        json.dump(model_x, f)

    with open(os.path.join(script_dir, './models/model_y2.json'), 'w+') as f:
        json.dump(model_y, f)

    with open(os.path.join(script_dir, './models/model_z2.json'), 'w+') as f:
        json.dump(model_z, f)

    with open(os.path.join(script_dir, './models/model_yaw2.json'), 'w+') as f:
        json.dump(model_yaw, f)

    with open(os.path.join(script_dir, './models/model_pitch2.json'), 'w+') as f:
        json.dump(model_pitch, f)
