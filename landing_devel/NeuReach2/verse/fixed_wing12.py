# Implement Algorithm 1 described in Paper

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

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

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

def apply_model(model, point):
    dim = model['dim']
    cc = model['coef_center_center']
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
        center_center = cc[0]*x + cc[1]
        radius = cr[0] + x*cr[1] + x**2*cr[2]
        return center_center, radius 
    else:
        x = point[0]
        y = point[1]
        center_center = cc[0]*x + cc[1]*y + cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        return center_center, radius
        
def get_vision_estimation(point: np.ndarray, models) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model(models[0], point)
    y_c, y_r = apply_model(models[1], point)
    z_c, z_r = apply_model(models[2], point)
    yaw_c, yaw_r = apply_model(models[3], point)
    pitch_c, pitch_r = apply_model(models[4], point)


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

def verify_step(point, M, computation_steps, time_steps, ref):
    # print(C_step, step, i, point)

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    estimate_low, estimate_high = get_vision_estimation(point, M)
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
def verify_step_remote(point, M, computation_steps, time_steps, ref):
    # print(C_step, step, i, point)
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    estimate_low, estimate_high = get_vision_estimation(point, M)
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

def compute_and_check(X_0, M, R):
    # x, y, z, yaw, pitch, v
    ray.init(num_cpus=12,log_to_driver=False)
    # state = np.array([
    #     [-3050.0, -20, 110.0, 0-0.01, -np.deg2rad(3)-0.01, 10-0.1], 
    #     [-3010.0, 20, 130.0, 0+0.01, -np.deg2rad(3)+0.01, 10+0.1]
    # ])
    state = X_0
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

    # Parameters
    num_sample = 100
    computation_steps = 0.1
    C_compute_step = 80
    C_num = 10
    parallel = True

    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    for C_step in range(C_num):
        # try:
            reachable_set = []
            for step in range(C_compute_step):
                print(">>>>>>>>>>>>>>>>", C_step, step)
                box = get_bounding_box(hull)
                state_low = box[0,:]
                state_high = box[1,:]

                reachable_set.append([np.insert(state_low, 0, step*computation_steps), np.insert(state_high, 0, step*computation_steps)])

                traces_list = []
                if step == 0:
                    # vertex_num = int(num_sample*0.05)
                    # sample_num = num_sample - vertex_num
                    # vertex_idxs = np.random.choice(hull.vertices, vertex_num)
                    vertex_sample = hull.points[hull.vertices,:]
                    # edge_sample = get_edge_samples(vertex_sample)
                    sample_sample = sample_point_poly(hull, num_sample)
                    samples = np.vstack((vertex_sample, sample_sample))
                else:
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
                        task_list.append(verify_step_remote.remote(point, M, computation_steps, time_steps, ref))
                    else:
                        print(C_step, step, i, point)
                        trace = verify_step(point, M, computation_steps, time_steps, ref)
                        traces_list.append(trace)

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
            
            # TODO: Check containment of C_set and R
            res = check_containment(C_set, R)
            if res == 'unsafe' or res == 'unknown':
                return res, C_list

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
        # except:
        #     break

    ray.shutdown()
    return 'safe', C_list


if __name__ == "__main__":
    
    