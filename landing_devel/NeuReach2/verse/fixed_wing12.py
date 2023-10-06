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
from get_all_models import get_all_models
from models.model_pitch2 import compute_model_pitch

import pickle 
import json 
import ray
from Rrect import R1

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

def apply_model_batch(model, point):
    dim = model['dim']
    cc = model['coef_center']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[:,0]
    elif dim == 'y':
        point = point[:,(0,1)]
    elif dim == 'z':
        point = point[:,(0,2)]
    elif dim == 'yaw':
        point = point[:,(0,3)]
    elif dim == 'pitch':
        point = point[:,(0,4)]

    if dim == 'x':
        x = point 
        center = cc[0]*x+cc[1]
        radius = cr[0]+x*cr[1]+x**2*cr[2]
        return center, radius
    elif dim == 'pitch' or dim == 'z':
        x = point[:,0]
        y = point[:,1]
        center = cc[0]*x + cc[1]*y +cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2]
        return center, radius
    else:
        x = point[:,0]
        y = point[:,1]
        center = cc[0]*x+cc[1]*y+cc[2]
        radius = cr[0]+x*cr[1]+y*cr[2]+x*y*cr[3]+x**2*cr[4]+y**2*cr[5]
        return center, radius

def get_vision_estimation_batch(point: np.ndarray, models) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model_batch(models[0], point)
    y_c, y_r = apply_model_batch(models[1], point)
    z_c, z_r = apply_model_batch(models[2], point)
    yaw_c, yaw_r = apply_model_batch(models[3], point)
    pitch_c, pitch_r = apply_model_batch(models[4], point)


    low = np.array([x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[:,5]]).T
    high = np.array([x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[:,5]]).T

    return low, high

def apply_model(model, point):
    dim = model['dim']
    cc = model['coef_center']
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
        return center_center, abs(radius) 
    elif dim == 'pitch' or dim == 'z':
        x = point[0]
        y = point[1]
        center = cc[0]*x + cc[1]*y +cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2]
        return center, abs(radius)
    else:
        x = point[0]
        y = point[1]
        center_center = cc[0]*x + cc[1]*y + cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        return center_center, abs(radius)
        
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

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step, M):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point, M)
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

def check_containment(post_rect, R, idx):
    # return 'safe'
    R_rect = R[idx]
    if (R_rect[0,0]<post_rect[0,1]<R_rect[1,0] and R_rect[0,0]<post_rect[1,1]<R_rect[1,0]) and \
        (R_rect[0,1]<post_rect[0,2]<R_rect[1,1] and R_rect[0,1]<post_rect[1,2]<R_rect[1,1]) and \
        (R_rect[0,2]<post_rect[0,3]<R_rect[1,2] and R_rect[0,2]<post_rect[1,3]<R_rect[1,2]):
        return 'safe'
    elif (
            (R_rect[0,0]<post_rect[0,1]<R_rect[1,0] or R_rect[0,0]<post_rect[1,1]<R_rect[1,0] or \
             post_rect[0,1]<R_rect[0,0]<post_rect[1,1] or post_rect[0,1]<R_rect[1,0]<post_rect[1,1]) and \
            (R_rect[0,1]<post_rect[0,2]<R_rect[1,1] or R_rect[0,1]<post_rect[1,2]<R_rect[1,1] or \
             post_rect[0,2]<R_rect[0,1]<post_rect[1,2] or post_rect[0,2]<R_rect[1,1]<post_rect[1,2]) and \
            (R_rect[0,2]<post_rect[0,3]<R_rect[1,2] or R_rect[0,2]<post_rect[1,3]<R_rect[1,2] or \
             post_rect[0,3]<R_rect[0,2]<post_rect[1,3] or post_rect[0,3]<R_rect[1,2]<post_rect[1,3])):
        return 'unknown'
    else:
        print(R_rect)
        print(post_rect)
        return 'unsafe'

def compute_and_check(X_0, M, R, depth=0):
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
    time_steps = 0.01
    
    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []

    for C_step in range(C_num):
        # try:
            reachable_set = []
            for step in range(C_compute_step):
                print(">>>>>>>>>>>>>>>>", depth, C_step, step)
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
            C_set[0,2] -= 1
            C_set[1,2] += 1
            C_set[0,3] -= 1
            C_set[1,3] += 1
            res = check_containment(C_set, R, C_step+1)
            if res == 'unsafe' or res == 'unknown':
                ray.shutdown()
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

def remove_data(data, E):
    # Remove data that are not in the range of E. 
    state_array, trace_array, E_array = data 
    idx = np.array([])
    for E_range in E:
        tmp = np.where((E_array[:,0]>E_range[0,0]) &\
                       (E_array[:,0]<E_range[1,0]) &\
                       (E_array[:,1]>E_range[0,1]) &\
                       (E_array[:,1]<E_range[1,1])
                       )[0]
        idx = np.concatenate((idx,tmp))
    idx = np.unique(idx).astype('int')
    return copy.deepcopy(state_array[idx,:]), copy.deepcopy(trace_array[idx,:]), copy.deepcopy(E_array[idx,:])

def computeContract(data, E):
    # Trim data according to the environments
    trimed_data = remove_data(data, E)
    M = get_all_models(trimed_data)
    return M

def refineState(X, idx):
    P1 = copy.deepcopy(X)
    P2 = copy.deepcopy(X)
    if idx%4==0 or idx%4==1:
        P1[1,1] = (P1[0,1] + P1[1,1])/2
        P2[0,1] = (P2[0,1] + P2[1,1])/2
    elif idx%4==2:
        P1[1,3] = (P1[0,3] + P1[1,3])/2
        P2[0,3] = (P2[0,3] + P2[1,3])/2
    elif idx%4==3:
        P1[1,0] = (P1[0,0] + P1[1,0])/2
        P2[0,0] = (P2[0,0] + P2[1,0])/2
    elif idx%4==4:
        P1[1,2] = (P1[0,2] + P1[1,2])/2
        P2[0,2] = (P2[0,2] + P2[1,2])/2
    elif idx%4==5:
        P1[1,4] = (P1[0,4] + P1[1,4])/2
        P2[0,4] = (P2[0,4] + P2[1,4])/2
    return P1, P2

def visualize_outlier(num_outlier_list, E, idx = 0):
    full_e = np.ones((20, 14))*(-1)
    for i in range(len(E)):
        E_part = E[i]
        idx1 = round((E_part[0,0]-0.2)/0.05)
        idx2 = round((E_part[0,1]-(-0.1))/0.05)
        full_e[idx1, idx2] = num_outlier_list[i]
    full_e = full_e/np.max(full_e)
    rgba_image = np.zeros((20, 14, 4))  # 4 channels: R, G, B, A
    rgba_image[..., :3] = plt.cm.viridis(full_e)[..., :3]  # Apply a colormap    
    mask = np.where(full_e<0)
    rgba_image[..., 3] = 1.0  # Set alpha to 1 (non-transparent)

    # Apply the mask to make some pixels transparent
    rgba_image[mask[0], mask[1], :3] = 1  # Set alpha to 0 (transparent) for masked pixels

    plt.imshow(rgba_image)
    plt.savefig(f'./{idx}.png')

def refineEnv(E_in, M, data, id = 0, vis = False):
    E = copy.deepcopy(E_in)
    state_array, trace_array, E_array = data 

    # # Compute bound for every data
    # xc, xr = apply_model_batch(M[0], state_array)
    # yc, yr = apply_model_batch(M[1], state_array)
    # zc, zr = apply_model_batch(M[2], state_array)
    # yawc, yawr = apply_model_batch(M[3], state_array)
    # pitchc, pitchr = apply_model_batch(M[4], state_array)
    
    # # Get the points that a furthest from center 
    # c = np.hstack((
    #     xc.reshape((-1,1)), 
    #     yc.reshape((-1,1)), 
    #     zc.reshape((-1,1)), 
    #     yawc.reshape((-1,1)), 
    #     pitchc.reshape((-1,1))
    # ))
    dist_array = np.linalg.norm(state_array[:,:5]-trace_array[:,:5], axis=1)

    # Get environmental parameters of those points 
    percentile = np.percentile(dist_array, 90)

    # Remove those environmental parameters from E
    idx_list = np.where(dist_array>percentile)[0]

    E = np.array(E)
    num_outlier_list = np.zeros(E.shape[0])
    for idx in idx_list:
        contains = np.where(
            (E[:,0,0]<E_array[idx,0]) & \
            (E_array[idx,0]<E[:,1,0]) & \
            (E[:,0,1]<E_array[idx,1]) & \
            (E_array[idx,1]<E[:,1,1])
        )[0]
        num_outlier_list[contains] += 1
    sorted_outlier = np.argsort(num_outlier_list)
    delete_outlier = sorted_outlier[-10:] 
    if vis:
        visualize_outlier(num_outlier_list, E, id)
    E = np.delete(E, delete_outlier, axis=0)
    return E 

def findM(X_0, E_0, R, data):
    E = E_0 
    M = computeContract(data, E)
    queue = [(X_0,0)]
    while len(queue) != 0:
        P, idx = queue.pop(0)
        res, C_list = compute_and_check(P, M, R, idx)
        if res == "unsafe":
            print("unsafe")
            return None, None, P, C_list
        elif res == "unknown":
            print("unknown")
            if False:
                P_1, P_2 = refineState(P, idx)
                queue.append((P_1, idx+1))
                queue.append((P_2, idx+1))
            else:
                queue.append((P, idx+1))
            E = refineEnv(E, M, data, True)
            if len(E) <= 10:
                return None, None, P, C_list
            M = computeContract(data, E)
    print("safe")
    return M, E, None, C_list

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

def partitionE(E):
    partition_list = []
    E1 = np.arange(E[0,0], E[1,0]+0.01, 0.05)
    E2 = np.arange(E[0,1], E[1,1]+0.01, 0.05)

    for i in range(len(E1)-1):
        for j in range(len(E2)-1):
            partition = np.array([
                [E1[i], E2[j]],
                [E1[i+1], E2[j+1]]
            ])
            partition_list.append(partition)

    return partition_list

def check_M(M, E, data):
    trimed_data = remove_data(data, E)
    state_array, trace_array, E_array = trimed_data
    
    total_num = len(state_array)
    total_contain = 0
    x_contain = 0
    y_contain = 0
    z_contain = 0
    yaw_contain = 0
    pitch_contain = 0
    for i in range(len(state_array)):
        lb, ub = get_vision_estimation(state_array[i,:], M)
        if all(lb[:5]<trace_array[i,:5]) and all(trace_array[i,:5] < ub[:5]):
            total_contain += 1 
        if lb[0] < trace_array[i,0] and trace_array[i,0] < ub[0]:
            x_contain += 1 
        if lb[1] < trace_array[i,1] and trace_array[i,1] < ub[1]:
            y_contain += 1 
        if lb[2] < trace_array[i,2] and trace_array[i,2] < ub[2]:
            z_contain += 1 
        if lb[3] < trace_array[i,3] and trace_array[i,3] < ub[3]:
            yaw_contain += 1 
        if lb[4] < trace_array[i,4] and trace_array[i,4] < ub[4]:
            pitch_contain += 1 
    print(total_contain/total_num)
    print(x_contain/total_num)
    print(y_contain/total_num)
    print(z_contain/total_num)
    print(yaw_contain/total_num)
    print(pitch_contain/total_num)

    total = len(state_array)
    contain = 0
    model_pitch = compute_model_pitch(state_array, trace_array, E_array, 0.8, 0.98)
    cc = model_pitch['coef_center']
    cr = model_pitch['coef_radius']
    for i in range(len(state_array)):
        c = cc[0]*state_array[i,0] + cc[1]*state_array[i,4] + cc[2]
        # c, r = apply_model(model_pitch, state_array[i,:])
        # radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        r = cr[0] + state_array[i,0]*cr[1] + state_array[i,4]*cr[2]
        if c-r < trace_array[i,4] < c+r:
            contain += 1
 
    print(contain/total)
    plt.plot(state_array[:,4], trace_array[:,4], 'r*')
    c, r = apply_model_batch(model_pitch, state_array)
    plt.plot(state_array[:,4], c-r, 'b*')
    plt.plot(state_array[:,4], c+r, 'b*')
    plt.show()

if __name__ == "__main__":
    
    X0 = np.array([
        [-3020.0, -5, 118.0, 0-0.001, -np.deg2rad(3)-0.001, 10-0.01], 
        [-3010.0, 5, 122.0, 0+0.001, -np.deg2rad(3)+0.001, 10+0.01]
    ])
    
    E = np.array([
        [0.2, -0.1],
        [1.2, 0.6]
    ])

    E = partitionE(E)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_train_exp1.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    data = pre_process_data(data)
    
    for i in range(10):
        E = refineEnv(E, None, data, i)
    M_out = computeContract(data, E)

    M_out, E_out, Part, C_list = findM(X0, E, R1, data)

    with open('tmp.pickle', 'wb+') as f:
        pickle.dump((M_out, E_out, C_list), f)

    refineEnv(E_out, M_out, data)

    # M = computeContract(data, E)
    # with open(os.path.join(script_dir, 'computed_model.pickle'), 'rb') as f:
    #     M_out, E_out, C_list = pickle.load(f)

    # check_M(M_out, E, data)
    # res, C_list = compute_and_check(X0, M_out, R1)

    # with open(os.path.join(script_dir, 'test.pickle'), 'wb+') as f:
    #     pickle.dump((M_out, E, None), f)

    # with open('computed_cone_ref.pickle','wb+') as f:
    #     pickle.dump(C_list, f)

    # with open('computed_cone_ref.pickle','rb') as f:
    #     C_list = pickle.load(f)

    computation_steps = 0.1
    time_steps = 0.01
    C_compute_step = 80
    C_num = 10

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

    # state = np.array([
    #     [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
    #     [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    # ])
    state = np.array([
        [-3020.0, -5, 118.0, 0-0.001, -np.deg2rad(3)-0.001, 10-0.01], 
        [-3010.0, 5, 122.0, 0+0.001, -np.deg2rad(3)+0.001, 10+0.01]
    ])
    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
    
    time_horizon = computation_steps*C_num*C_compute_step

    # fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    # # fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    # script_path = os.path.realpath(os.path.dirname(__file__))
    # aircraft = FixedWingAgent3("a1")
    # fixed_wing_scenario.add_agent(aircraft)

    # for i in range(20):
    #     init_point = sample_point(state[0,:], state[1,:])
    #     init_ref = copy.deepcopy(ref)
    #     trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, computation_steps, time_steps, M_out)
    #     trace = np.array(trace)
    #     plt.figure(0)
    #     plt.plot(trace[:,1], trace[:,2], 'r')
    #     plt.figure(1)
    #     plt.plot(trace[:,0], trace[:,1], 'r')
    #     plt.figure(2)
    #     plt.plot(trace[:,0], trace[:,2], 'r')
    #     plt.figure(3)
    #     plt.plot(trace[:,0], trace[:,3], 'r')
    #     plt.figure(4)
    #     plt.plot(trace[:,0], trace[:,4], 'r')
    #     plt.figure(5)
    #     plt.plot(trace[:,0], trace[:,5], 'r')
    #     plt.figure(6)
    #     plt.plot(trace[:,0], trace[:,6], 'r')

    plt.show()
       
