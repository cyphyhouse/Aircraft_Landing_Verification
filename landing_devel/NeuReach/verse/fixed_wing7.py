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
from datetime import datetime 

import pickle 

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
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

def get_bounding_box(hull: scipy.spatial.ConvexHull) -> np.ndarray:
    vertices = hull.points[hull.vertices, :]
    lower_bound = np.min(vertices, axis=0)
    upper_bound = np.max(vertices, axis=0)
    return np.vstack((lower_bound, upper_bound))

def in_hull(point: np.ndarray, hull:scipy.spatial.ConvexHull) -> bool:
    tmp = hull
    if not isinstance(hull, scipy.spatial.Delaunay):
        tmp = scipy.spatial.Delaunay(hull.points[hull.vertices,:], qhull_options='Qt Qbb Qc Qz Qx Q12 QbB')
    
    return tmp.find_simplex(point) >= 0

def sample_point_poly(hull: scipy.spatial.ConvexHull, n: int) -> np.ndarray:
    vertices = hull.points[hull.vertices,:]
    weights = np.random.uniform(0,1,vertices.shape[0])
    weights = weights/np.sum(weights)
    start_point = np.zeros(vertices.shape[1])
    for i in range(vertices.shape[1]):
        start_point[i] = np.sum(vertices[:,i]*weights)
    # return start_point

    sampled_point = []
    for i in range(n):
        vertex_idx = np.random.choice(hull.vertices)
        vertex = hull.points[vertex_idx, :]
        offset = vertex - start_point 
        start_point = start_point + np.random.uniform(0,1)*offset 
        sampled_point.append(start_point)

    return np.array(sampled_point)

# def sample_point_poly(hull: scipy.spatial.ConvexHull):
#     box = get_bounding_box(hull)
#     point = np.random.uniform(box[0,:], box[1,:])
#     while not in_hull(point, hull):
#         point = np.random.uniform(box[0,:], box[1,:])
#     return point 
    # verts = hull.points[hull.vertices,:]
    # vert_mean = np.mean(verts, axis=0)
    # weights = np.random.uniform(0,1,(verts.shape[0]))
    # weights = weights/np.sum(weights)
    # # point = np.sum((verts-vert_mean)*weights, axis=0)
    # res = verts[0,:]*weights[0]
    # for i in range(1, len(weights)):
    #     res += verts[i,:]*weights[i]
    # return res

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

# def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
#     vertex_list = []
#     for analysis_tree in trace_list:
#         rect_low = analysis_tree.nodes[0].trace['a1'][-2][1:7]
#         rect_high = analysis_tree.nodes[0].trace['a1'][-1][1:7]
#         # tmp = [
#         #     [rect_low[0], rect_high[0]],
#         #     [rect_low[1], rect_high[1]],
#         #     [rect_low[2], rect_high[2]],
#         #     [rect_low[3], rect_high[3]],
#         #     [rect_low[4], rect_high[4]],
#         #     [rect_low[5], rect_high[5]],
#         # ]
#         vertex_list.append(rect_low)
#         vertex_list.append(rect_high)
#     vertices = np.array(vertex_list)
#     # hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB')
    
#     return vertices

def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
    vertex_list = []
    for analysis_tree in trace_list:
        rect_low = analysis_tree.nodes[0].trace['a1'][-2][1:7]
        rect_high = analysis_tree.nodes[0].trace['a1'][-1][1:7]
        tmp = [
            [rect_low[0], rect_high[0]],
            [rect_low[1], rect_high[1]],
            [rect_low[2], rect_high[2]],
            [rect_low[3], rect_high[3]],
            [rect_low[4], rect_high[4]],
            [rect_low[5], rect_high[5]],
        ]
        vertex_list.append(rect_low)
        vertex_list.append(rect_high)
    # vertices = np.array(vertex_list)
    vertices = []
    for vertex in vertex_list:
        away = True
        for i in range(len(vertices)):
            if np.linalg.norm(np.array(vertex)-np.array(vertices[i]))<0.01:
                away = False
                break 
        if away:
            vertices.append(vertex)

    vertices = np.array(vertices)
    hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB Q12 Qc')    
    return hull

def run_sim_random(
        state: np.ndarray, 
        ref: np.ndarray, 
        scenario: Scenario,
        time_horizon: float,
        time_step: float
    ):
    upper, lower = state[0,:], state[1,:]
    point = sample_point(lower, upper)
    estimate_low, estimate_high = get_vision_estimation(point)
    estimation = sample_point(estimate_low, estimate_high)

    time_list = np.arange(0, time_horizon+time_step/2, time_step)

    init= np.concatenate((point, estimation, ref))

    tmp = init_low[1:7]

    traj = np.insert(tmp, 0, 0)

    for t in time_list[1:]:
        scenario.set_init(
            [init],
            [
                (FixedWingMode.Normal,)
            ],
        )
        traces = scenario.simulate(time_step, time_step)
        point = traces.nodes[0].trace['a1'][-1][1:7]
        tmp = np.insert(point, 0, t)
        traj = np.vstack((traj, tmp))
        estimate_low, estimate_high = get_vision_estimation(point)
        estimation = sample_point(estimate_low, estimate_high)
        next_ref = run_ref(init[12:],time_step)
        init = np.concatenate((point, estimation, ref))

    return traj

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point)
        estimate_point = sample_point(estimate_lower, estimate_upper)
        init = np.concatenate((point, estimate_point, ref))
        scenario.set_init(
            [[init]],
            [(FixedWingMode.Normal,)]
        )
        res = scenario.simulate(computation_step, time_step)
        trace = res.nodes[0].trace['a1']
        point = trace[-1,1:7]
        traj.append(np.insert(point, 0, t))
        ref = run_ref(ref, time_step)
    return traj

if __name__ == "__main__":
    
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    state = np.array([
        [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
        [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    ])
    tmp = [
        [state[0,0], state[1,0]],
        [state[0,1], state[1,1]],
        [state[0,2], state[1,2]],
        [state[0,3], state[1,3]],
        [state[0,4], state[1,4]],
        [state[0,5], state[1,5]],
    ]
    vertices = np.array(list(itertools.product(*tmp)))
    hull = scipy.spatial.ConvexHull(vertices)
    # state_low = state[0,:]
    # state_high = state[1,:]
    num_dim = state.shape[1]

    # Parameters
    num_sample = 2000
    computation_steps = 0.1
    time_steps = 0.01
    C_compute_step = 50
    C_num = 1

    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    for C_step in range(C_num):
        reachable_set = []
        for step in range(C_compute_step):
            box = get_bounding_box(hull)
            state_low = box[0,:]
            state_high = box[1,:]

            reachable_set.append([np.insert(state_low, 0, step*computation_steps), np.insert(state_high, 0, step*computation_steps)])

            traces_list = []
            point_list = []
            point_idx_list = []
            
            if step == 37:
                print('stop')
            
            if hull.vertices.shape[0]<num_sample:
                # vertex_num = int(num_sample*0.05)
                # sample_num = num_sample - vertex_num
                # vertex_idxs = np.random.choice(hull.vertices, vertex_num)
                vertex_sample = hull.points[hull.vertices,:]
                sample_sample = sample_point_poly(hull, 100)
                samples = np.vstack((vertex_sample, sample_sample))
            else:
                vertex_num = int(num_sample*0.5)
                sample_num = num_sample - vertex_num
                vertex_idxs = np.random.choice(hull.vertices, vertex_num, replace=False)
                vertex_sample = hull.points[vertex_idxs,:]
                sample_sample = sample_point_poly(hull, sample_num)
                samples = np.vstack((vertex_sample, sample_sample))
                

            for i in range(samples.shape[0]):
                point = samples[i,:]
                print(step, i, point)

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
                traces = fixed_wing_scenario.verify(computation_steps, time_steps, params={'bloating_method':'GLOBAL'})
                traces_list.append(traces)

            hull = get_next_poly(traces_list)

            # state_low = next_low 
            # state_high = next_high 
            ref = run_ref(ref, computation_steps)
        
        next_init = get_bounding_box(hull)
        # last_rect = reachable_set[-1]
        # next_init = np.array(last_rect)[:,1:]
        C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))
        C_list.append(C_set)

        tmp = [
            [next_init[0,0], next_init[1,0]],
            [next_init[0,1], next_init[1,1]],
            [next_init[0,2], next_init[1,2]],
            [next_init[0,3], next_init[1,3]],
            [next_init[0,4], next_init[1,4]],
            [next_init[0,5], next_init[1,5]],
        ]
        vertices = np.array(list(itertools.product(*tmp)))
        hull = scipy.spatial.ConvexHull(vertices)

    for C_rect in C_list:
        # rect_low = C_rect[0]
        # rect_high = C_rect[1]

        low = C_rect[0]
        high = C_rect[1]
        step_time = low[0]*C_compute_step*computation_steps
        plt.figure(0)
        plt.plot(
            [low[1], high[1], high[1], low[1], low[1]], 
            [low[2], low[2], high[2], high[2], low[2]],
            'b'
        )
        plt.figure(1)
        plt.plot(
            [step_time, step_time], [low[1], high[1]],
            'b'
        )
        plt.figure(2)
        plt.plot(
            [step_time, step_time], [low[2], high[2]],
            'b'
        )
        plt.figure(3)
        plt.plot(
            [step_time, step_time], [low[3], high[3]],
            'b'
        )
        plt.figure(4)
        plt.plot(
            [step_time, step_time], [low[4], high[4]],
            'b'
        )
        plt.figure(5)
        plt.plot(
            [step_time, step_time], [low[5], high[5]],
            'b'
        )
        plt.figure(6)
        plt.plot(
            [step_time, step_time], [low[6], high[6]],
            'b'
        )

    state = np.array([
        [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
        [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    ])
    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
    time_horizon = computation_steps*C_num*C_compute_step

    for i in range(100):
        init_point = sample_point(state[0,:], state[1,:])
        init_ref = copy.deepcopy(ref)
        trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, computation_steps, time_steps)
        trace = np.array(trace)
        plt.figure(0)
        plt.plot(trace[:,1], trace[:,2], 'r')
        plt.figure(1)
        plt.plot(trace[:,0], trace[:,1], 'r')
        plt.figure(2)
        plt.plot(trace[:,0], trace[:,2], 'r')
        plt.figure(3)
        plt.plot(trace[:,0], trace[:,3], 'r')
        plt.figure(4)
        plt.plot(trace[:,0], trace[:,4], 'r')
        plt.figure(5)
        plt.plot(trace[:,0], trace[:,5], 'r')
        plt.figure(6)
        plt.plot(trace[:,0], trace[:,6], 'r')

    plt.show()
        

    # start_time = datetime.now()
    # time_str = start_time.strftime("%m-%d_%H-%M-%S")

    # with open(f'reachable_set_{time_str}.pickle', 'wb+') as f:
    #     pickle.dump(reachable_set, f)

    # plt.figure(0)
    # plt.figure(1)
    # plt.figure(2)
    # for rectangle in reachable_set:
    #     low = rectangle[0]
    #     high = rectangle[1]
    #     plt.figure(0)
    #     plt.plot(
    #         [low[1], high[1], high[1], low[1], low[1]], 
    #         [low[2], low[2], high[2], high[2], low[2]],
    #         'b'
    #     )
    #     plt.figure(1)
    #     plt.plot(
    #         [low[0], high[0]], [low[1], high[1]],
    #         'b'
    #     )
    #     plt.figure(2)
    #     plt.plot(
    #         [low[0], high[0]], [low[2], high[2]],
    #         'b'
    #     )
    #     plt.figure(3)
    #     plt.plot(
    #         [low[0], high[0]], [low[3], high[3]],
    #         'b'
    #     )
    #     plt.figure(4)
    #     plt.plot(
    #         [low[0], high[0]], [low[4], high[4]],
    #         'b'
    #     )
    #     plt.figure(5)
    #     plt.plot(
    #         [low[0], high[0]], [low[5], high[5]],
    #         'b'
    #     )
    #     plt.figure(6)
    #     plt.plot(
    #         [low[0], high[0]], [low[6], high[6]],
    #         'b'
    #     )
    # plt.show()

