import pickle 
import numpy as np 
import torch
from model import get_model_rect2, get_model_rect, get_model_rect3
from enum import Enum, auto
from typing import Tuple
import os 

from verse.plotter.plotter2D import *

import mpl_toolkits.mplot3d as a3
import polytope as pc
import pypoman as ppm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt 
import copy

from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig

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
    yaw_r = np.array([0.001])
    yaw_c = np.array([point[3]])

    input_tensor = torch.FloatTensor([point[(0,4),]], device='cpu')
    pitch_r = forward_pitch_r(input_tensor).detach().numpy()
    pitch_c = forward_pitch_c(input_tensor).detach().numpy()
    pitch_r = np.abs(np.reshape(pitch_r, (-1)))
    pitch_c = np.reshape(pitch_c, (-1))
    pitch_r = np.array([0.001])
    pitch_c = np.array([point[4]])

    low = np.concatenate((x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[5:]))
    high = np.concatenate((x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[5:]))

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
    with open('computed_cone_old.pickle', 'rb') as f:
        C_list = pickle.load(f)

    C_list_truncate = C_list[:12]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-3050, -3010)
    ax.set_ylim(-20, 20)
    ax.set_zlim(110, 130)

    for i in range(len(C_list_truncate)):
        rect = C_list[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k')

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    

    state = np.array([
        [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
        [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    ])
    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
    time_horizon = 0.1*((len(C_list_truncate)-1)*80+1)

    # for i in range(10):
    #     init_point = sample_point(state[0,:], state[1,:])
    #     init_ref = copy.deepcopy(ref)
    #     trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, 0.1, 0.01)
    #     trace = np.array(trace)
    #     ax.plot(trace[:,1], trace[:,2], trace[:,3], linewidth=1, color='g')
    #     ax.scatter(trace[::80,1], trace[::80,2], trace[::80,3], marker='x', color='m', s=30)

    # with open('./src/landing_devel/NeuReach/verse/vcs_sim.pickle','rb') as f:
    with open('vcs_sim.pickle','rb') as f:
        vcs_sim_trajectories = pickle.load(f)

    for traj in vcs_sim_trajectories:
        traj = np.array(traj)[:(len(C_list)-1)*80+1,:]
        ax.plot(traj[:,1], traj[:,2], traj[:,3], linewidth=1, color='b')
        ax.scatter(traj[::80,1], traj[::80,2], traj[::80,3], marker='x', color='m', s=30)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()