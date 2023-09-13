from verse.plotter.plotter2D import *
# from fixed_wing_agent import FixedWingAgent
# from fixed_wing_agent2 import AircraftTrackingAgent
from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 
# from model import get_model_rect2, get_model_rect, get_model_rect3
import torch
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt 
import polytope as pc
import itertools
import scipy.spatial
from datetime import datetime 
from verse.analysis.verifier import ReachabilityMethod

import pickle 
import json 
import ray

script_dir = os.path.realpath(os.path.dirname(__file__))

model_x_name = '../models/model_x.json'
model_y_name = '../models/model_y.json'
model_z_name = '../models/model_z.json'
model_yaw_name = '../models/model_yaw.json'
model_pitch_name = '../models/model_pitch.json'

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

# model_x_r_name = "checkpoint_x_r_06-27_23-32-30_471.pth.tar"
# model_x_c_name = "checkpoint_x_c_06-27_23-32-30_471.pth.tar"
# model_y_r_name = 'checkpoint_y_r_07-11_15-43-23_40.pth.tar'
# model_y_c_name = 'checkpoint_y_c_07-11_15-43-23_40.pth.tar'
# model_z_r_name = 'checkpoint_z_r_07-11_14-50-20_24.pth.tar'
# model_z_c_name = 'checkpoint_z_c_07-11_14-50-20_24.pth.tar'
# model_roll_r_name = 'checkpoint_roll_r_07-11_16-53-35_45.pth.tar'
# model_roll_c_name = 'checkpoint_roll_c_07-11_16-53-35_45.pth.tar'
# model_pitch_r_name = 'checkpoint_pitch_r_07-11_16-54-45_44.pth.tar'
# model_pitch_c_name = 'checkpoint_pitch_c_07-11_16-54-45_44.pth.tar'
# model_yaw_r_name = 'checkpoint_yaw_r_07-11_17-03-31_180.pth.tar'
# model_yaw_c_name = 'checkpoint_yaw_c_07-11_17-03-31_180.pth.tar'

# model_x_r, forward_x_r = get_model_rect2(1,1,64,64,64)
# model_x_c, forward_x_c = get_model_rect(1,1,64,64)
# model_x_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_x_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_x_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_x_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_x_r.eval()
# model_x_c.eval()

# model_y_r, forward_y_r = get_model_rect2(2,1,64,64,64)
# model_y_c, forward_y_c = get_model_rect(2,1,64,64)
# model_y_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_y_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_y_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_y_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_y_r.eval()
# model_y_c.eval()

# model_z_r, forward_z_r = get_model_rect2(2,1,64,64,64)
# model_z_c, forward_z_c = get_model_rect(2,1,64,64)
# model_z_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_z_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_z_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_z_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_z_r.eval()
# model_z_c.eval()

# model_roll_r, forward_roll_r = get_model_rect2(2,1,64,64,64)
# model_roll_c, forward_roll_c = get_model_rect(2,1,64,64)
# model_roll_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_roll_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_roll_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_roll_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_roll_r.eval()
# model_roll_c.eval()

# model_pitch_r, forward_pitch_r = get_model_rect2(2,1,64,64,64)
# model_pitch_c, forward_pitch_c = get_model_rect(2,1,64,64)
# model_pitch_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_pitch_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_pitch_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_pitch_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_pitch_r.eval()
# model_pitch_c.eval()

# model_yaw_r, forward_yaw_r = get_model_rect2(2,1,64,64,64)
# model_yaw_c, forward_yaw_c = get_model_rect(2,1,64,64)
# model_yaw_r.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_yaw_r_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_yaw_c.load_state_dict(torch.load(os.path.join(script_dir, f'../log/{model_yaw_c_name}'), map_location=torch.device('cpu'))['state_dict'])
# model_yaw_r.eval()
# model_yaw_c.eval()

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
        point = point[(0,3),]
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
        
def get_vision_estimation(point: np.ndarray, Ec: np.ndarray, Er: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model(model_x, point, Ec, Er)
    y_c, y_r = apply_model(model_y, point, Ec, Er)
    z_c, z_r = apply_model(model_z, point, Ec, Er)
    yaw_c, yaw_r = apply_model(model_yaw, point, Ec, Er)
    pitch_c, pitch_r = apply_model(model_pitch, point, Ec, Er)


    low = np.array([x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[5]])
    high = np.array([x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[5]])

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

def get_next_poly(trace_list) -> scipy.spatial.ConvexHull:
    vertex_list = []
    sample_vertex = np.zeros((0,6))
    for trace in trace_list:
        # trace = pickle.loads(tmp)
        rect_low = trace[-2,1:7]
        rect_high = trace[-1,1:7]
        # tmp = [
        #     [rect_low[0], rect_high[0]],
        #     [rect_low[1], rect_high[1]],
        #     [rect_low[2], rect_high[2]],
        #     [rect_low[3], rect_high[3]],
        #     [rect_low[4], rect_high[4]],
        #     [rect_low[5], rect_high[5]],
        # ]
        # vertices = np.array(list(itertools.product(*tmp)))
        sample = np.random.uniform(rect_low, rect_high)
        sample_vertex = np.vstack((sample_vertex, sample))
        vertex_list.append(rect_low)
        vertex_list.append(rect_high)

    # sample_idx  = np.random.choice(sample_vertex.shape[0], sample_vertex.shape[0]-23, replace = False)
    # sample_vertex = sample_vertex[sample_idx, :]
    sample_vertex = sample_vertex[:64,:]
    vertices = []
    for vertex in vertex_list:
        away = True
        for i in range(len(vertices)):
            if np.linalg.norm(np.array(vertex)-np.array(vertices[i]))<0.05:
                away = False
                break 
        if away:
            vertices.append(vertex)

    vertices = np.array(vertices)
    hull = scipy.spatial.ConvexHull(vertices, qhull_options='Qx Qt QbB Q12 Qc')    
    return hull, sample_vertex

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step, Ec, Er):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point, Ec, Er)
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
        ref = run_ref(ref, computation_step)
    return traj

def verify_step(point, Ec, Er, ref):
    # print(C_step, step, i, point)

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    estimate_low, estimate_high = get_vision_estimation(point, Ec, Er)
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
    # tmp = pickle.dumps(np.array(traces))
    # tmp = pickle.dumps(traces.root.trace['a1'])
    return np.array(traces.root.trace['a1'])

@ray.remote
def verify_step_remote(point, Ec, Er, ref):
    # print(C_step, step, i, point)
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    estimate_low, estimate_high = get_vision_estimation(point, Ec, Er)
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
    # tmp = pickle.dumps(np.array(traces))
    # tmp = pickle.dumps(traces.root.trace['a1'])
    return np.array(traces.root.trace['a1'])

if __name__ == "__main__":
    
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    # fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    ray.init(num_cpus=12,log_to_driver=False)
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
    num_sample = 150
    computation_steps = 0.1
    time_steps = 0.01
    C_compute_step = 80
    C_num = 10
    parallel = True
    Ec = [0.85] 
    Er = [0.05]

    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    for C_step in range(C_num):
        try:
            reachable_set = []
            for step in range(C_compute_step):
                print(">>>>>>>>>>>>>>>>", C_step, step)
                box = get_bounding_box(hull)
                state_low = box[0,:]
                state_high = box[1,:]

                reachable_set.append([np.insert(state_low, 0, step*computation_steps), np.insert(state_high, 0, step*computation_steps)])

                traces_list = []
                point_list = []
                point_idx_list = []
                
                # if step == 37:
                #     print('stop')

                if step == 0:
                    # vertex_num = int(num_sample*0.05)
                    # sample_num = num_sample - vertex_num
                    # vertex_idxs = np.random.choice(hull.vertices, vertex_num)
                    vertex_sample = hull.points[hull.vertices,:]
                    sample_sample = sample_point_poly(hull, num_sample)
                    samples = np.vstack((vertex_sample, sample_sample))
                else:
                    # vertex_num = int(num_sample*0.5)
                    # sample_num = num_sample - vertex_num
                    # vertex_idxs = np.random.choice(hull.vertices, vertex_num, replace=False)
                    # vertex_sample = hull.points[vertex_idxs,:]
                    # sample_sample = sample_point_poly(hull, sample_num)
                    # samples = np.vstack((vertex_sample, sample_sample))

                    sample_sample = sample_point_poly(hull, num_sample)
                    samples = np.vstack((vertex_sample, sample_sample))
                    # samples = vertex_sample
                
                point_idx = np.argmax(hull.points[:,1])
                samples = np.vstack((samples, hull.points[point_idx,:]))
                # samples = sample_point_poly(hull, num_sample)
                
                point_idx = np.argmax(hull.points[:,0])
                samples = np.vstack((samples, hull.points[point_idx,:]))
                point_idx = np.argmin(hull.points[:,0])
                samples = np.vstack((samples, hull.points[point_idx,:]))

                task_list = []
                traces_list = []
                for i in range(samples.shape[0]):

                    point = samples[i,:]
                    
                    if parallel:
                        task_list.append(verify_step_remote.remote(point, Ec, Er, ref))
                    else:
                        print(C_step, step, i, point)
                        trace = verify_step(point, Ec, Er, ref)
                        traces_list.append(trace)

                    # estimate_low, estimate_high = get_vision_estimation(point, [0.85], [0.35])

                    # init_low = np.concatenate((point, estimate_low, ref))
                    # init_high = np.concatenate((point, estimate_high, ref))
                    # init = np.vstack((init_low, init_high))       

                    # fixed_wing_scenario.set_init(
                    #     [init],
                    #     [
                    #         (FixedWingMode.Normal,)
                    #     ],
                    # )
                    # # TODO: WE should be able to initialize each of the balls separately
                    # # this may be the cause for the VisibleDeprecationWarning
                    # # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
                    # # "-2 \leq myball1.x + myball2.x \leq 5"
                    # traces = fixed_wing_scenario.verify(computation_steps, time_steps, params={'bloating_method':'GLOBAL'})
                    # traces_list.append(traces)

                if parallel:
                    traces_list = ray.get(task_list)
                # hull2, vertex_sample2 = get_next_poly(traces_list)
                hull, vertex_sample = get_next_poly(traces_list)
                # box1 = get_bounding_box(hull)
                # box2 = get_bounding_box(hull2)
                # if (box1 != box2).any():
                #     print('stop')
                # plt.figure(6)
                # plt.plot([step*0.1, step*0.1],[box[0,-1],box[1,-1]],'g')
                # state_low = next_low 
                # state_high = next_high 
                ref = run_ref(ref, computation_steps)
            
            next_init = get_bounding_box(hull)
            # last_rect = reachable_set[-1]
            # next_init = np.array(last_rect)[:,1:]
            C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))
            C_list.append(C_set)

            with open('computed_cone_085_05.pickle','wb+') as f:
                pickle.dump(C_list, f)

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
        except:
            break

    ray.shutdown()

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

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    # fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)

    for i in range(20):
        init_point = sample_point(state[0,:], state[1,:])
        init_ref = copy.deepcopy(ref)
        trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, computation_steps, time_steps, Ec, Er)
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
       
