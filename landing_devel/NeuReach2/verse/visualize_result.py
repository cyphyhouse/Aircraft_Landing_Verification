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

if __name__ == "__main__":
    # with open('./src/landing_devel/NeuReach/verse/computed_cone.pickle','rb') as f:
    with open('computed_cone_35.pickle', 'rb') as f:
        C_list_35 = pickle.load(f)

    C_list_35_truncate = C_list_35[:12]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-3050, -3010)
    ax.set_ylim(-20, 20)
    ax.set_zlim(110, 130)

    for i in range(len(C_list_35_truncate)):
        rect = C_list_35[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k')

    with open('computed_cone_25.pickle', 'rb') as f:
        C_list_25 = pickle.load(f)
    C_list_25_truncate = C_list_25[:12]
    for i in range(len(C_list_25_truncate)):
        rect = C_list_25[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k', color='b')

    with open('computed_cone_15.pickle', 'rb') as f:
        C_list_15 = pickle.load(f)
    C_list_15_truncate = C_list_15[:12]
    for i in range(len(C_list_15_truncate)):
        rect = C_list_15[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k', color='g')

    with open('computed_cone_05.pickle', 'rb') as f:
        C_list_05 = pickle.load(f)
    C_list_05_truncate = C_list_05[:12]
    for i in range(len(C_list_05_truncate)):
        rect = C_list_05[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k', color='y')

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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir,'vcs_sim.pickle'),'rb') as f:
        vcs_sim_trajectories = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_estimate.pickle'), 'rb') as f:
        vcs_sim_estimate = pickle.load(f)
    with open(os.path.join(script_dir,'vcs_init.pickle'),'rb') as f:
        vcs_sim_init = pickle.load(f)

    for i in range(len(vcs_sim_trajectories)):
        traj = np.array(vcs_sim_trajectories[i])
        ax.plot(traj[:,1], traj[:,2], traj[:,3], linewidth=1, color='b')
        ax.scatter(traj[::80,1], traj[::80,2], traj[::80,3], marker='x', color='m', s=30)

    for i in range(len(C_list_35)):
        rect = C_list_35[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,1]<traj[i*80][1]<rect[1,1] and \
                rect[0,2]<traj[i*80][2]<rect[1,2] and \
                rect[0,3]<traj[i*80][3]<rect[1,3] and \
                rect[0,4]<traj[i*80][4]<rect[1,4] and \
                rect[0,5]<traj[i*80][5]<rect[1,5]
            ):
                print(35, i, j, vcs_sim_init[j])
                # break

    for i in range(len(C_list_25)):
        rect = C_list_25[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,1]<traj[i*80][1]<rect[1,1] and \
                rect[0,2]<traj[i*80][2]<rect[1,2] and \
                rect[0,3]<traj[i*80][3]<rect[1,3] and \
                rect[0,4]<traj[i*80][4]<rect[1,4] and \
                rect[0,5]<traj[i*80][5]<rect[1,5]
            ):
                print(25, i, j, vcs_sim_init[j])
                # break

    for i in range(len(C_list_15)):
        rect = C_list_15[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,1]<traj[i*80][1]<rect[1,1] and \
                rect[0,2]<traj[i*80][2]<rect[1,2] and \
                rect[0,3]<traj[i*80][3]<rect[1,3] and \
                rect[0,4]<traj[i*80][4]<rect[1,4] and \
                rect[0,5]<traj[i*80][5]<rect[1,5]
            ):
                print(15, i, j, vcs_sim_init[j])
                # break

    for i in range(len(C_list_05)):
        rect = C_list_05[i]
        for j, traj in enumerate(vcs_sim_trajectories):
            if not (
                rect[0,1]<traj[i*80][1]<rect[1,1] and \
                rect[0,2]<traj[i*80][2]<rect[1,2] and \
                rect[0,3]<traj[i*80][3]<rect[1,3] and \
                rect[0,4]<traj[i*80][4]<rect[1,4] and \
                rect[0,5]<traj[i*80][5]<rect[1,5]
            ):
                print(5, i, j, vcs_sim_init[j])
                # break


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()