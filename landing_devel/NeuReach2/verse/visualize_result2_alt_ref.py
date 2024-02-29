import pickle 
import numpy as np 
import torch
from enum import Enum, auto
from typing import Tuple
import os 
import json 

from verse.plotter.plotter2D import *

import mpl_toolkits.mplot3d as a3
import polytope as pc
import pypoman as ppm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt 
import copy

from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig

import os 
from Rrect import R1
from fixed_wing12 import pre_process_data, get_vision_estimation_batch

model_radius_decay = lambda r, r_max: (1/np.sqrt(r_max))*np.sqrt(r) # Input to this function is the radius of environmental parameters

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

def plot_polytope_3d(A, b, ax = None, edgecolor = 'k', color = 'red', trans = 0.2):
    verts = np.array(ppm.compute_polytope_vertices(A, b))
    # compute the triangles that make up the convex hull of the data points
    hull = ConvexHull(verts)
    triangles = [verts[s] for s in hull.simplices]
    # combine co-planar triangles into a single face
    faces = Faces(triangles, sig_dig=1).simplify()
    # plot
    if ax == None:
        ax = a3.Axes3D(plt.figure())

    pc = a3.art3d.Poly3DCollection(faces,
                                    facecolor=color,
                                    edgecolor=edgecolor, alpha=trans)
    ax.add_collection3d(pc)
    # define view
    yllim, ytlim = ax.get_ylim()
    xllim, xtlim = ax.get_xlim()
    zllim, ztlim = ax.get_zlim()
    x = verts[:,0]
    x = np.append(x, [xllim+1, xtlim-1])
    y = verts[:,1]
    y = np.append(y, [yllim+1, ytlim-1])
    z = verts[:,2]
    z = np.append(z, [zllim+1, ztlim-1])
    # print(np.min(x)-1, np.max(x)+1, np.min(y)-1, np.max(y)+1, np.min(z)-1, np.max(z)+1)
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_zlim(np.min(z)-1, np.max(z)+1)

script_dir = os.path.realpath(os.path.dirname(__file__))

model_x_name = './models/model_x2.json'
model_y_name = './models/model_y2.json'
model_z_name = './models/model_z2.json'
model_yaw_name = './models/model_yaw2.json'
model_pitch_name = './models/model_pitch2.json'

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

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

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
        return center_center, radius 
    if dim == 'pitch' or dim == 'z':
        x = point[0]
        y = point[1]
        center_center = cc[0]*x + cc[1]*y + cc[2]
        radius = cr[0] + x*cr[1] + y*cr[2]
        return center_center, abs(radius)
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

def sample_point(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.random.uniform(low, high) 

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
        ref = run_ref(ref, computation_step)
    return traj

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    fig = plt.figure(0)
    ax = plt.axes(projection='3d')
    ax.set_xlim(-3050, -3010)
    ax.set_ylim(-20, 20)
    ax.set_zlim(110, 130)

    fn = os.path.join(script_dir, './exp1_res_safe_alt_ref.pickle')
    with open(fn, 'rb') as f:
        M, E, C_list = pickle.load(f)
    C_list_truncate = C_list[:12]
    for i in range(len(C_list_truncate)):
        rect = C_list[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k')

    for i in range(len(R1)):
        rect =  R1[i]

        pos_rect = rect[:, 0:3]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k', color='b')

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    

    # state = np.array([
    #     [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
    #     [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    # ])
    # ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
    # time_horizon = 0.1*((len(C_list_truncate)-1)*80+1)

    # for i in range(10):
    #     init_point = sample_point(state[0,:], state[1,:])
    #     init_ref = copy.deepcopy(ref)
    #     trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, 0.1, 0.01)
    #     trace = np.array(trace)
    #     ax.plot(trace[:,1], trace[:,2], trace[:,3], linewidth=1, color='g')
    #     ax.scatter(trace[::80,1], trace[::80,2], trace[::80,3], marker='x', color='m', s=30)

    # with open('./src/landing_devel/NeuReach/verse/vcs_sim.pickle','rb') as f:
    with open(os.path.join(script_dir,'vcs_sim_exp1_safe_alt_ref.pickle'),'rb') as f:
        vcs_sim_trajectories = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_estimate_exp1_safe_alt_ref.pickle'), 'rb') as f:
        vcs_sim_estimate = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_init_exp1_safe_alt_ref.pickle'),'rb') as f:
        vcs_sim_init = pickle.load(f)

    total = 0
    not_in = 0
    not_in_0 = 0
    not_in_1 = 0
    not_in_2 = 0
    not_in_3 = 0
    not_in_4 = 0
    for j in range(len(vcs_sim_trajectories)):
        traj = vcs_sim_trajectories[j]
        est = vcs_sim_estimate[j]
        for k in range(len(traj)-1):
            total += 1
            point = traj[k][1:]
            lb, ub = get_vision_estimation(point, M)
            est_point = est[k]
            if any((lb[:5]>est_point[:5]) | (est_point[:5]>ub[:5])):
                tmp = (lb[:5]>est_point[:5]) | (est_point[:5]>ub[:5])
                print(tmp)
                not_in += 1 
                if tmp[0]:
                    not_in_0 += 1
                if tmp[1]:
                    not_in_1 += 1
                if tmp[2]:
                    not_in_2 += 1
                if tmp[3]:
                    not_in_3 += 1
                if tmp[4]:
                    not_in_4 += 1
    print(total, not_in)
    print(not_in_0,not_in_1,not_in_2,not_in_3,not_in_4)

    unsafe_idx_list = []
    for i in range(len(C_list)):
        rect = C_list[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,1]<traj[i*80][1]<rect[1,1] and \
                rect[0,2]<traj[i*80][2]<rect[1,2] and \
                rect[0,3]<traj[i*80][3]<rect[1,3] and \
                rect[0,4]<traj[i*80][4]<rect[1,4] and \
                rect[0,5]<traj[i*80][5]<rect[1,5]
            ):
                print(rect[0,1]<traj[i*80][1]<rect[1,1]) 
                print(rect[0,2]<traj[i*80][2]<rect[1,2]) 
                print(rect[0,3]<traj[i*80][3]<rect[1,3]) 
                print(rect[0,4]<traj[i*80][4]<rect[1,4]) 
                print(rect[0,5]<traj[i*80][5]<rect[1,5])
                print(35, i, j, vcs_sim_init[j])
                if j not in unsafe_idx_list:
                    unsafe_idx_list.append(j)

    for idx in range(len(vcs_sim_trajectories)):
        traj = np.array(vcs_sim_trajectories[idx])
        # if idx not in unsafe_idx_list:
        ax.plot(traj[:800,1], traj[:800,2], traj[:800,3], linewidth=1, color='b')
        # else:
        ax.scatter(traj[:801:80,1], traj[:801:80,2], traj[:801:80,3], marker='x', color='m', s=30)

    with open(os.path.join(script_dir,'vcs_sim_exp1_safecomp_alt_ref.pickle'),'rb') as f:
        vcs_sim_trajectories = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_estimate_exp1_safecomp_alt_ref.pickle'), 'rb') as f:
        vcs_sim_estimate = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_init_exp1_safecomp_alt_ref.pickle'),'rb') as f:
        vcs_sim_init = pickle.load(f)

    unsafe_idx_list = []
    for i in range(len(C_list)):
        rect = C_list[i]
        rect = R1[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,0]<traj[i*80][1]<rect[1,0] and \
                rect[0,1]<traj[i*80][2]<rect[1,1] and \
                rect[0,2]<traj[i*80][3]<rect[1,2]
            ):
                if j not in unsafe_idx_list:
                    unsafe_idx_list.append(j)

    for idx in range(len(vcs_sim_trajectories)):
        traj = np.array(vcs_sim_trajectories[idx])
        # if idx not in unsafe_idx_list:
        #     ax.plot(traj[:800,1], traj[:800,2], traj[:800,3], linewidth=1, color='#ff7f0e')
        # else:
        ax.plot(traj[:800,1], traj[:800,2], traj[:800,3], linewidth=1, color='r')
        ax.scatter(traj[:801:80,1], traj[:801:80,2], traj[:801:80,3], marker='x', color='m', s=30)
    print(len(unsafe_idx_list))
    ax.set_xlabel('x', fontsize = 22)
    ax.set_ylabel('y', fontsize = 22)
    ax.set_zlabel('z', fontsize = 22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    # tmp_sim_trajectories = vcs_sim_trajectories[:10] + vcs_sim_trajectories[19:29]
    # with open(os.path.join(script_dir,'vcs_sim.pickle'),'rb') as f:
    #     pickle.dump(tmp_sim_trajectories, f)
    # tmp_sim_estimate = vcs_sim_estimate[:10] + vcs_sim_estimate[19:29]
    # with open(os.path.join(script_dir,'vcs_estimate.pickle'), 'rb') as f:
    #     pickle.dump(tmp_sim_estimate, f)
    # tmp_sim_init = vcs_sim_init[:10] + vcs_sim_init[19:29]
    # with open(os.path.join(script_dir,'vcs_init.pickle'),'rb') as f:
    #     pickle.dump(tmp_sim_init, f)

    with open(os.path.join(script_dir, '../data_train_exp1.pickle'), 'rb') as f:
        data = pickle.load(f)
    data = pre_process_data(data)
    state_array, trace_array, E_array = data

    miss_array = np.zeros(len(E))
    total_array = np.zeros(len(E))
    for i, Ep in enumerate(E):
        in_part = np.where(
            (Ep[0,0]<E_array[:,0]) & \
            (E_array[:,0]<Ep[1,0]) & \
            (Ep[0,1]<E_array[:,1]) & \
            (E_array[:,1]<Ep[1,1])
        )[0]
        total_array[i] = len(in_part)
        state_contain = state_array[in_part]
        trace_contain = trace_array[in_part]
        E_contain = E_array[in_part]
        # for j in range(len(in_part)):
        lb, ub = get_vision_estimation_batch(state_contain, M)
        tmp = np.where(np.any((lb[:,:5]>trace_contain[:,:5]) | (trace_contain[:,:5]>ub[:,:5]), axis=1))[0]
        miss_array[i] = len(tmp)

    accuracy_array = (total_array-miss_array)/total_array
    print((total_array-miss_array).sum()/total_array.sum())

    plt.figure(1)
    # e_plane = np.zeros((20,14))
    # for Ep in E:
    #     idx1 = int(round((Ep[0,0]-0.2)/0.05))
    #     idx2 = int(round((Ep[0,1]+0.1)/0.05))
    #     e_plane[idx1, idx2] = 1
    # plt.imshow(e_plane)
    full_e = np.ones((20, 14))*(-1)
    min_acc = float('inf')
    min_acc_e1 = 0
    min_acc_e2 = 0
    for i in range(len(E)):
        E_part = E[i]
        idx1 = round((E_part[0,0]-0.2)/0.05)
        idx2 = round((E_part[0,1]-(-0.1))/0.05)
        full_e[idx1, idx2] = accuracy_array[i]
        if accuracy_array[i]!=0 and accuracy_array[i]<min_acc:
            min_acc = accuracy_array[i]
            min_acc_e1 = E_part[0,0]
            min_acc_e2 = E_part[0,1]
    print(min_acc, min_acc_e1, min_acc_e2)
    # full_e = full_e/np.max(full_e)
    rgba_image = np.zeros((20, 14, 4))  # 4 channels: R, G, B, A
    rgba_image[..., :3] = plt.cm.viridis(full_e)[..., :3]  # Apply a colormap    
    mask = np.where(full_e<0)
    rgba_image[..., 3] = 1.0  # Set alpha to 1 (non-transparent)

    # Apply the mask to make some pixels transparent
    rgba_image[mask[0], mask[1], :3] = 1  # Set alpha to 0 (transparent) for masked pixels

    plt.imshow(rgba_image)

    ax = plt.gca()
    ax.set_xticks(np.round(np.arange(0,16,2)-0.5,1))
    ax.set_yticks(np.round(np.arange(0,22,2)-0.5,1))
    ax.set_xticklabels(np.round(np.arange(-0.1, 0.65, 0.1),2), fontsize=14)
    ax.set_yticklabels(np.round(np.arange(0.2, 1.25, 0.1),2), fontsize=14)
    plt.xlabel('Spotlight yaw', fontsize=16)
    plt.ylabel('Ambient light intensity', fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    # plt.xticks(list(range(0,14,2)), np.round(np.arange(-0.05,0.6,0.1),2))
    # plt.yticks(list(range(0,20,2)), np.round(np.arange(0.25, 1.2, 0.1),2))
    plt.show()