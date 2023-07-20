from verse.plotter.plotter2D import *
from fixed_wing_agent import FixedWingAgent
from fixed_wing_agent2 import AircraftTrackingAgent
from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 
from model import get_model_rect2, get_model_rect, get_model_rect3
import torch
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt 
import polytope as pc
import itertools
import scipy.spatial
import time

script_dir = os.path.realpath(os.path.dirname(__file__))

model_x_r_name = "checkpoint_x_r_06-27_23-32-30_471.pth.tar"
model_x_c_name = "checkpoint_x_c_06-27_23-32-30_471.pth.tar"
model_y_r_name = 'checkpoint_y_r_07-11_15-43-23_40.pth.tar'
model_y_c_name = 'checkpoint_y_c_07-11_15-43-23_40.pth.tar'
model_z_r_name = 'checkpoint_z_r_07-11_14-50-20_24.pth.tar'
model_z_c_name = 'checkpoint_z_c_07-11_14-50-20_24.pth.tar'
model_roll_r_name = 'checkpoint_roll_r_07-11_16-53-35_45.pth.tar'
model_roll_c_name = 'checkpoint_roll_c_07-11_16-53-35_45.pth.tar'
model_pitch_r_name = 'checkpoint_pitch_r_07-11_16-54-45_44.pth.tar'
model_pitch_c_name = 'checkpoint_pitch_c_07-11_16-54-45_44.pth.tar'
model_yaw_r_name = 'checkpoint_yaw_r_07-11_17-03-31_180.pth.tar'
model_yaw_c_name = 'checkpoint_yaw_c_07-11_17-03-31_180.pth.tar'

model_x_r, forward_x_r = get_model_rect2(1,1,64,64,64)
model_x_c, forward_x_c = get_model_rect(1,1,64,64)
model_x_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_x_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_x_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_x_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_x_r.eval()
model_x_c.eval()

model_y_r, forward_y_r = get_model_rect2(2,1,64,64,64)
model_y_c, forward_y_c = get_model_rect(2,1,64,64)
model_y_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_y_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_y_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_y_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_y_r.eval()
model_y_c.eval()

model_z_r, forward_z_r = get_model_rect2(2,1,64,64,64)
model_z_c, forward_z_c = get_model_rect(2,1,64,64)
model_z_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_z_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_z_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_z_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_z_r.eval()
model_z_c.eval()

model_roll_r, forward_roll_r = get_model_rect2(2,1,64,64,64)
model_roll_c, forward_roll_c = get_model_rect(2,1,64,64)
model_roll_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_roll_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_roll_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_roll_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_roll_r.eval()
model_roll_c.eval()

model_pitch_r, forward_pitch_r = get_model_rect2(2,1,64,64,64)
model_pitch_c, forward_pitch_c = get_model_rect(2,1,64,64)
model_pitch_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_pitch_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_pitch_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_pitch_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_pitch_r.eval()
model_pitch_c.eval()

model_yaw_r, forward_yaw_r = get_model_rect2(2,1,64,64,64)
model_yaw_c, forward_yaw_c = get_model_rect(2,1,64,64)
model_yaw_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_yaw_r_name}'), map_location=torch.device('cpu'))['state_dict'])
model_yaw_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_yaw_c_name}'), map_location=torch.device('cpu'))['state_dict'])
model_yaw_r.eval()
model_yaw_c.eval()

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)


# class TrackMode(Enum):
#     Lane0 = auto()
#     #For now this is a dummy notion of Lane


class State:
    """Defines the state variables of the model
    Both discrete and continuous variables
    """

    x: float
    y: float 
    z: float 
    yaw: float 
    pitch: float 
    v: float
    x_est: float 
    y_est: float 
    z_est: float 
    yaw_est: float 
    pitch_est: float 
    v_est: float
    x_ref: float 
    y_ref: float 
    z_ref: float 
    yaw_ref: float 
    pitch_ref: float 
    v_ref: float
    mode: FixedWingMode

    def __init__(self, x, y, z, yaw, pitch, v, x_est, y_est, z_est, yaw_est, pitch_est, v_est, x_ref, y_ref, z_ref, yaw_ref, pitch_ref, v_ref, mode: FixedWingMode):
        pass


def decisionLogic(ego: State):
    """Computes the possible mode transitions"""
    output = copy.deepcopy(ego)
    return output

def sample_point(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.random.uniform(low, high) 

def get_vision_estimation(point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    input_tensor = torch.FloatTensor([[point[0]]], device='cpu')
    x_r = forward_x_r(input_tensor).detach().numpy()
    x_c = forward_x_c(input_tensor).detach().numpy()
    x_r = np.abs(np.reshape(x_r, (-1)))
    x_c = np.reshape(x_c, (-1))
    
    input_tensor = torch.FloatTensor([point[(0,1),]], device='cpu')
    y_r = forward_y_r(input_tensor).detach().numpy()
    y_c = forward_y_c(input_tensor).detach().numpy()
    y_r = np.abs(np.reshape(y_r, (-1)))
    y_c = np.reshape(y_c, (-1))

    input_tensor = torch.FloatTensor([point[(0,2),]], device='cpu')
    z_r = forward_z_r(input_tensor).detach().numpy()
    z_c = forward_z_c(input_tensor).detach().numpy()
    z_r = np.abs(np.reshape(z_r, (-1)))
    z_c = np.reshape(z_c, (-1))

    input_tensor = torch.FloatTensor([point[(0,3),]], device='cpu')
    yaw_r = forward_yaw_r(input_tensor).detach().numpy()
    yaw_c = forward_yaw_c(input_tensor).detach().numpy()
    yaw_r = np.abs(np.reshape(yaw_r, (-1)))
    yaw_c = np.reshape(yaw_c, (-1))

    input_tensor = torch.FloatTensor([point[(0,4),]], device='cpu')
    pitch_r = forward_pitch_r(input_tensor).detach().numpy()
    pitch_c = forward_pitch_c(input_tensor).detach().numpy()
    pitch_r = np.abs(np.reshape(pitch_r, (-1)))
    pitch_c = np.reshape(pitch_c, (-1))

    low = np.concatenate((x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[5:]))
    high = np.concatenate((x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[5:]))

    return low, high

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x*time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

def get_bounding_box(hull: scipy.spatial.ConvexHull) -> np.ndarray:
    # vertices = hull.points[hull.vertices, :]
    lower_bound = np.min(hull, axis=0)
    upper_bound = np.max(hull, axis=0)
    return np.vstack((lower_bound, upper_bound))

# def in_hull(point: np.ndarray, hull: scipy.spatial.ConvexHull) -> bool:
#     A = hull.equations[:,:6]
#     b = hull.equations[:,6:]
#     point = np.reshape(point, (-1,1))
#     if np.all(A@point<=b):
#         return True 
#     else:
#         return False

def in_hull(point: np.ndarray, hull:scipy.spatial.ConvexHull) -> bool:
    tmp = hull
    if not isinstance(hull, scipy.spatial.Delaunay):
        tmp = scipy.spatial.Delaunay(hull.points[hull.vertices,:], qhull_options='Qt Qbb Qc Qz Qx Q12 QbB')
    
    return tmp.find_simplex(point) >= 0

def sample_point_poly(hull: scipy.spatial.ConvexHull):
    # box = get_bounding_box(hull)
    # point = np.random.uniform(box[0,:], box[1,:])
    # while not in_hull(point, hull):
    #     point = np.random.uniform(box[0,:], box[1,:])
    # return point 
    verts = hull.points[hull.vertices,:]
    vert_mean = np.mean(verts, axis=0)
    weights = np.random.uniform(0,1,(verts.shape[0]))
    weights = weights/np.sum(weights)
    # point = np.sum((verts-vert_mean)*weights, axis=0)
    res = verts[0,:]*weights[0]
    for i in range(1, len(weights)):
        res += verts[i,:]*weights[i]
    return res

# def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
#     vertex_list = []
#     for analysis_tree in trace_list:
#         rect_low = analysis_tree.nodes[0].trace['a1'][-2][1:7]
#         rect_high = analysis_tree.nodes[0].trace['a1'][-1][1:7]
#         tmp = [
#             [rect_low[0], rect_high[0]],
#             [rect_low[1], rect_high[1]],
#             [rect_low[2], rect_high[2]],
#             [rect_low[3], rect_high[3]],
#             [rect_low[4], rect_high[4]],
#             [rect_low[5], rect_high[5]],
#         ]
#         vertex_list += list(itertools.product(*tmp))
#     vertices = np.array(vertex_list)
#     hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB')
    
#     return hull

def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
    vertex_list = []
    for analysis_tree in trace_list:
        rect_low = analysis_tree.nodes[0].trace['a1'][-2][1:7]
        rect_high = analysis_tree.nodes[0].trace['a1'][-1][1:7]
        # tmp = [
        #     [rect_low[0], rect_high[0]],
        #     [rect_low[1], rect_high[1]],
        #     [rect_low[2], rect_high[2]],
        #     [rect_low[3], rect_high[3]],
        #     [rect_low[4], rect_high[4]],
        #     [rect_low[5], rect_high[5]],
        # ]
        vertex_list.append(rect_low)
        vertex_list.append(rect_high)
    vertices = np.array(vertex_list)
    # hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB')
    
    return vertices


if __name__ == "__main__":
    
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    state = np.array([
        [-3050.0, -20, 110.0, 0-0.00001, -np.deg2rad(3)-0.00001, 0-0.00001], 
        [-3010.0, 20, 130.0, 0+0.00001, -np.deg2rad(3)+0.00001, 0+0.00001]
    ])
    tmp = [
        [state[0,0], state[1,0]],
        [state[0,1], state[1,1]],
        [state[0,2], state[1,2]],
        [state[0,3], state[1,3]],
        [state[0,4], state[1,4]],
        [state[0,5], state[1,5]],
    ]
    hull = np.array(list(itertools.product(*tmp)))
    # hull = scipy.spatial.ConvexHull(vertices)
    # state_low = state[0,:]
    # state_high = state[1,:]
    num_dim = state.shape[1]

    # Parameters
    num_sample = 20
    computation_steps = 0.05
    time_steps = 0.01

    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    reachable_set = []
    point_idx_list_list = []
    point_list_list = []

    for step in range(100):

        box = get_bounding_box(hull)
        state_low = box[0,:]
        state_high = box[1,:]

        reachable_set.append([np.insert(state_low, 0, step*computation_steps), np.insert(state_high, 0, step*computation_steps)])

        traces_list = []
        point_list = []
        point_idx_list = []
        
        if step == 37:
            print('stop')

        for i in range(num_sample):
            print(step, i)
            # if i<10:
            np.random.seed(int((time.time()-1689780000)*10000))
            point_idx = np.random.choice(hull.shape[0])
            point = hull[point_idx,:]
            point_list.append(point)
            point_idx_list.append(point_idx)
            # else:
            #     point = sample_point_poly(hull)

            estimate_low, estimate_high = get_vision_estimation(point)

            init_low = np.concatenate((point, estimate_low, ref))
            init_high = np.concatenate((point, estimate_high, ref))
            init = np.vstack((init_low, init_high))       

            fixed_wing_scenario.set_init(
                [init],
                [
                    (FixedWingMode.Normal,)
                ],
            )
            # TODO: WE should be able to initialize each of the balls separately
            # this may be the cause for the VisibleDeprecationWarning
            # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
            # "-2 \leq myball1.x + myball2.x \leq 5"
            traces = fixed_wing_scenario.verify(computation_steps, time_steps)
            traces_list.append(traces)

        point_idx_list_list.append(point_idx_list)
        point_list_list.append(point_list)
        # Combine traces to get next init set 
        # next_low = np.array([float('inf')]*num_dim)
        # next_high = np.array([-float('inf')]*num_dim)
        
        # next_ref = []
        # for i in range(len(traces_list)):
        #     trace = traces_list[i].nodes[0].trace['a1']
        #     trace_low = trace[-2][1:7]
        #     trace_high = trace[-1][1:7]
        #     next_low = np.minimum(trace_low, next_low)
        #     next_high = np.maximum(trace_high, next_high)
        #     # next_ref = trace[-1,13:]
        hull = get_next_poly(traces_list)

        # state_low = next_low 
        # state_high = next_high 

        ref = run_ref(ref, computation_steps)

    plt.figure(0)
    plt.figure(1)
    plt.figure(2)
    for rectangle in reachable_set:
        low = rectangle[0]
        high = rectangle[1]
        plt.figure(0)
        plt.plot(
            [low[1], high[1], high[1], low[1], low[1]], 
            [low[2], low[2], high[2], high[2], low[2]],
            'b'
        )
        plt.figure(1)
        plt.plot(
            [low[0], high[0]], [low[1], high[1]],
            'b'
        )
        plt.figure(2)
        plt.plot(
            [low[0], high[0]], [low[2], high[2]],
            'b'
        )
        plt.figure(3)
        plt.plot(
            [low[0], high[0]], [low[3], high[3]],
            'b'
        )
        plt.figure(4)
        plt.plot(
            [low[0], high[0]], [low[4], high[4]],
            'b'
        )
        plt.figure(5)
        plt.plot(
            [low[0], high[0]], [low[5], high[5]],
            'b'
        )
        plt.figure(6)
        plt.plot(
            [low[0], high[0]], [low[6], high[6]],
            'b'
        )
    plt.show()

