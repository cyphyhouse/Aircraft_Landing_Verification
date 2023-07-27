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
    yaw_r = np.array([0.0001])
    yaw_c = np.array([point[3]])

    input_tensor = torch.FloatTensor([point[(0,4),]], device='cpu')
    pitch_r = forward_pitch_r(input_tensor).detach().numpy()
    pitch_c = forward_pitch_c(input_tensor).detach().numpy()
    pitch_r = np.abs(np.reshape(pitch_r, (-1)))
    pitch_c = np.reshape(pitch_c, (-1))
    pitch_r = np.array([0.0001])
    pitch_c = np.array([point[4]])

    low = np.concatenate((x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[5:]))
    high = np.concatenate((x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[5:]))

    return low, high

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])


if __name__ == "__main__":
    
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    state = np.array([
        [-3050.0, -20, 110.0, 0, -np.deg2rad(3), 10], 
        [-3010.0, 20, 130.0, 0, -np.deg2rad(3), 10]
    ])
    state_low = state[0,:]
    state_high = state[1,:]
    num_dim = state.shape[1]

    # Parameters
    num_sample = 5
    computation_steps = 0.1
    time_steps = 0.01


    for i in range(20):
        ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
        point = sample_point(state_low, state_high)

        traj = [np.insert(point, 0, 0)]
        for step in range(1500):
            est_lower, est_upper = get_vision_estimation(point)
            est_point = sample_point(est_lower, est_upper)
            init = np.concatenate((point, est_point, ref))

            fixed_wing_scenario.set_init(
                [[init]],
                [
                    (FixedWingMode.Normal,)
                ],
            )

            res = fixed_wing_scenario.simulate(computation_steps, time_steps)
            lstate = res.nodes[0].trace['a1'][-1,1:7]
            traj.append(np.insert(lstate, 0, (step+1)*computation_steps))
            ref = run_ref(ref, computation_steps)
            point = lstate

        traj = np.array(traj)
        plt.figure(0)
        plt.plot(traj[:,1], traj[:,2], 'b')
        plt.figure(1)
        plt.plot(traj[:,0], traj[:,1], 'b')
        plt.figure(2)
        plt.plot(traj[:,0], traj[:,2], 'b')
        plt.figure(3)
        plt.plot(traj[:,0], traj[:,3], 'b')
        plt.figure(4)
        plt.plot(traj[:,0], traj[:,4], 'b')
        plt.figure(5)
        plt.plot(traj[:,0], traj[:,5], 'b')
        plt.figure(6)
        plt.plot(traj[:,0], traj[:,6], 'b')
    plt.show()