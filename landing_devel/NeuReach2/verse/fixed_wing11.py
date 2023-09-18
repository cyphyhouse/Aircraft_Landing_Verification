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
from verse.analysis.analysis_tree import AnalysisTree

import pickle 
import json 

from typing import List 

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

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point, [0.85], [0.35])
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

def verify_step(point, Ec, Er):
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
    return traces

def get_partitions(hull: List[np.ndarray])->List[np.ndarray]:
    # Given a list of rectangles,
    # Get the list of corresponding partitions
    bounding_box = get_bounding_box_partitions(hull)
    x_partition_size = int((bounding_box[1,0]-bounding_box[0,0])/10)+1
    y_partition_size = int((bounding_box[1,1]-bounding_box[0,1])/2)+1
    z_partition_size = int((bounding_box[1,2]-bounding_box[0,2])/5)+1
    yaw_partition_size = int((bounding_box[1,3]-bounding_box[0,3])/0.005)+1
    pitch_partition_size = int((bounding_box[1,4]-bounding_box[0,4])/0.005)+1
    v_partition_size = int((bounding_box[1,5]-bounding_box[0,5])/0.01)+1
    
    x_range = np.linspace(bounding_box[0,0], bounding_box[1,0], x_partition_size)
    y_range = np.linspace(bounding_box[0,1], bounding_box[1,1], y_partition_size)
    z_range = np.linspace(bounding_box[0,2], bounding_box[1,2], z_partition_size)
    yaw_range = np.linspace(bounding_box[0,3], bounding_box[1,3], yaw_partition_size)
    pitch_range = np.linspace(bounding_box[0,4], bounding_box[1,4], pitch_partition_size)
    v_range = np.linspace(bounding_box[0,5], bounding_box[1,5], v_partition_size)

    used_partitions = []
    partition_map = np.zeros((
        len(x_range)-1,
        len(y_range)-1,
        len(z_range)-1,
        len(yaw_range)-1,
        len(pitch_range)-1,
        len(v_range)-1,
    ))                 
    for partition in hull:
        x_indices_list = []
        for i in range(len(x_range)-1):
            if (partition[0,0]>=x_range[i] and partition[0,0]<=x_range[i+1]) or \
                (partition[1,0]>=x_range[i] and partition[1,0]<=x_range[i+1]) or \
                (partition[0,0]<=x_range[i] and partition[1,0]>=x_range[i+1]):
                x_indices_list.append(i)
        
        y_indices_list = []
        for i in range(len(y_range)-1):
            if (partition[0,1]>=y_range[i] and partition[0,1]<=y_range[i+1]) or \
                (partition[1,1]>=y_range[i] and partition[1,1]<=y_range[i+1]) or \
                (partition[0,1]<=y_range[i] and partition[1,1]>=y_range[i+1]):
                y_indices_list.append(i)

        z_indices_list = []
        for i in range(len(z_range)-1):
            if (partition[0,2]>=z_range[i] and partition[0,2]<=z_range[i+1]) or \
                (partition[1,2]>=z_range[i] and partition[1,2]<=z_range[i+1]) or \
                (partition[0,2]<=z_range[i] and partition[1,2]>=z_range[i+1]):
                z_indices_list.append(i)

        yaw_indices_list = []
        for i in range(len(yaw_range)-1):
            if (partition[0,3]>=yaw_range[i] and partition[0,3]<=yaw_range[i+1]) or \
                (partition[1,3]>=yaw_range[i] and partition[1,3]<=yaw_range[i+1]) or \
                (partition[0,3]<=yaw_range[i] and partition[1,3]>=yaw_range[i+1]):
                yaw_indices_list.append(i)

        pitch_indices_list = []
        for i in range(len(pitch_range)-1):
            if (partition[0,4]>=pitch_range[i] and partition[0,4]<=pitch_range[i+1]) or \
                (partition[1,4]>=pitch_range[i] and partition[1,4]<=pitch_range[i+1]) or \
                (partition[0,4]<=pitch_range[i] and partition[1,4]>=pitch_range[i+1]):
                pitch_indices_list.append(i)

        v_indices_list = []
        for i in range(len(v_range)-1):
            if (partition[0,5]>=v_range[i] and partition[0,5]<=v_range[i+1]) or \
                (partition[1,5]>=v_range[i] and partition[1,5]<=v_range[i+1]) or \
                (partition[0,5]<=v_range[i] and partition[1,5]>=v_range[i+1]):
                v_indices_list.append(i)

        new_partitions = list(itertools.product(x_indices_list, y_indices_list, z_indices_list, yaw_indices_list, pitch_indices_list, v_indices_list))

        for npar in new_partitions:
            if partition_map[npar]!=1:
                partition_map[npar]==1
                part = np.array([
                    [
                        x_range[npar[0]], 
                        y_range[npar[1]], 
                        z_range[npar[2]], 
                        yaw_range[npar[3]], 
                        pitch_range[npar[4]], 
                        v_range[npar[5]]
                    ],
                    [
                        x_range[npar[0]+1], 
                        y_range[npar[1]+1], 
                        z_range[npar[2]+1], 
                        yaw_range[npar[3]+1], 
                        pitch_range[npar[4]+1], 
                        v_range[npar[5]+1]
                    ],
                ])
                used_partitions.append(part)
    return used_partitions

def apply_model_partition(model, partition, Ec, Er):
    dim = model['dim']
    point_list = []
    point = partition[0,:]
    point_list.append(point)
    point = partition[1,:]
    point_list.append(point)
    if dim == 'x':
        pass
    elif dim == 'y':
        point = copy.deepcopy(partition[0,:])
        point[1] = partition[1,1]
        point_list.append(point)
        point = copy.deepcopy(partition[1,:])
        point[1] = partition[0,1]
        point_list.append(point)
    elif dim == 'z':
        point = copy.deepcopy(partition[0,:])
        point[2] = partition[1,2]
        point_list.append(point)
        point = copy.deepcopy(partition[1,:])
        point[2] = partition[0,2]
        point_list.append(point)
    elif dim == 'yaw':
        point = copy.deepcopy(partition[0,:])
        point[3] = partition[1,3]
        point_list.append(point)
        point = copy.deepcopy(partition[1,:])
        point[3] = partition[0,3]
        point_list.append(point)
    elif dim == 'pitch':
        point = copy.deepcopy(partition[0,:])
        point[4] = partition[1,4]
        point_list.append(point)
        point = copy.deepcopy(partition[1,:])
        point[4] = partition[0,4]
        point_list.append(point)

    max_r = 0 
    max_c = -float('inf')
    min_c = float('inf')

    for point in point_list:
        c, r = apply_model(model, point, Ec, Er)
        if c>max_c:
            max_c = c 
        if c<min_c:
            min_c = c 
        if r > max_r:
            max_r = r 
    return min_c-max_r, max_c+max_r 

def get_vision_estimation_partition(partition: np.ndarray, Ec: List[float], Er: List[float])->Tuple[np.ndarray, np.ndarray]:
    # Given a partition and a range of environmental parameters defined by Ec and Er
    # Get the range of possible percepted states 
    x_est_low, x_est_high = apply_model_partition(model_x, partition, Ec, Er)
    y_est_low, y_est_high = apply_model_partition(model_y, partition, Ec, Er)
    z_est_low, z_est_high = apply_model_partition(model_z, partition, Ec, Er)
    yaw_est_low, yaw_est_high = apply_model_partition(model_yaw, partition, Ec, Er)
    pitch_est_low, pitch_est_high = apply_model_partition(model_pitch, partition, Ec, Er)
    
    low = np.array([
        x_est_low,
        y_est_low,
        z_est_low,
        yaw_est_low,
        pitch_est_low,
        partition[0,-1]
    ])
    high = np.array([
        x_est_high,
        y_est_high,
        z_est_high,
        yaw_est_high,
        pitch_est_high,
        partition[1,-1]
    ])

    return low, high

def get_next_init(traces_list: List[AnalysisTree])->List[np.ndarray]:
    # Given a list of analysis trees,
    # Export the last rectangle in each reachable set
    res_list = []
    for trace in traces_list:
        low = trace.nodes[0].trace['a1'][-2][1:7]
        high = trace.nodes[0].trace['a1'][-1][1:7]
        res_list.append(np.array([low, high]))
    return res_list

def get_bounding_box_partitions(hull: List[np.ndarray]) -> np.ndarray:
    # Given a list of rectangles
    # Get a bounding box that enlose all rectangles
    hull_array = np.array(hull)
    hull_array = np.reshape(hull_array, (-1,hull_array.shape[2]))
    ub = np.max(hull_array, axis=0)
    lb = np.min(hull_array, axis=0)
    return np.vstack((lb, ub))

if __name__ == "__main__":
    
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.DRYVR_DISC)) 
    # fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    state = np.array([
        [-3050.0, -20, 110.0, 0-0.01, -np.deg2rad(3)-0.01, 10-0.01], 
        [-3010.0, 20, 130.0, 0+0.01, -np.deg2rad(3)+0.01, 10+0.01]
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
    num_sample = 200
    computation_steps = 0.1
    time_steps = 0.01
    C_compute_step = 80
    C_num = 15

    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    C_list = [np.hstack((np.array([[0],[0]]),state))]
    # point_idx_list_list = []
    # point_list_list = []
    hull = [state]

    for C_step in range(C_num):
        # try:
            reachable_set = []
            for step in range(C_compute_step):
                partitions = get_partitions(hull)
                traces_list = []
                for i in range(len(partitions)):
                    partition = partitions[i]
                    print(C_step, step, i, partition)
                    estimate_low, estimate_high = get_vision_estimation_partition(copy.deepcopy(partition), [0.85], [0.35])
                    
                    init_low = np.concatenate((partition[0,:], estimate_low, ref))
                    init_high = np.concatenate((partition[1,:], estimate_high, ref))

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

                hull = get_next_init(traces_list)

                ref = run_ref(ref, computation_steps)

                
            next_init = get_bounding_box_partitions(hull)
            # last_rect = reachable_set[-1]
            # next_init = np.array(last_rect)[:,1:]
            C_set = np.hstack((np.array([[C_step+1],[C_step+1]]), next_init))
            C_list.append(C_set)

            # with open('computed_cone_large.pickle','wb+') as f:
            #     pickle.dump(C_list, f)

            # tmp = [
            #     [next_init[0,0], next_init[1,0]],
            #     [next_init[0,1], next_init[1,1]],
            #     [next_init[0,2], next_init[1,2]],
            #     [next_init[0,3], next_init[1,3]],
            #     [next_init[0,4], next_init[1,4]],
            #     [next_init[0,5], next_init[1,5]],
            # ]
            # vertices = np.array(list(itertools.product(*tmp)))
            # hull = scipy.spatial.ConvexHull(vertices)
            hull = next_init
        # except:
        #     break

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

    for i in range(20):
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
       
