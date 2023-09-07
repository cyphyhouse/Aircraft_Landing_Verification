import numpy as np 
from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import matplotlib.pyplot as plt 

import pickle 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, 'vcs_sim_new.pickle'),'rb') as f:
    traj_list = pickle.load(f)
prev_traj = np.array(traj_list[0])
with open(os.path.join(script_dir, 'vcs_estimate.pickle'),'rb') as f:
    estimate_traj_list = pickle.load(f)
est_traj = np.array(estimate_traj_list[0])

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

def vision_estimation_playback(point, t):
    idx = round((t-0.1)/0.1)
    print(np.linalg.norm(point-prev_traj[idx,1:7]))
    est_point = est_traj[idx,:]
    return est_point 

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        # estimate_point = vision_estimation_playback(point, t)
        estimate_point = point
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
    init = prev_traj[0,1:7]
    init_ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    fixed_wing_controller = os.path.join(script_dir, './fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)

    traj = run_vision_sim(fixed_wing_scenario, init, init_ref, 20, 0.1, 0.01)

    traj = np.array(traj)
    plt.figure(0)
    plt.plot(traj[:,0], traj[:,1],'b')
    plt.plot(traj[:-1,0], est_traj[:,0],'r')
    plt.figure(1)
    plt.plot(traj[:,0], traj[:,2],'b')
    plt.plot(traj[:-1,0], est_traj[:,1],'r')
    plt.figure(2)
    plt.plot(traj[:,0], traj[:,3],'b')
    plt.plot(traj[:-1,0], est_traj[:,2],'r')
    plt.figure(3)
    plt.plot(traj[:,0], traj[:,4],'b')
    plt.plot(traj[:-1,0], est_traj[:,3],'r')
    plt.figure(4)
    plt.plot(traj[:,0], traj[:,5],'b')
    plt.plot(traj[:-1,0], est_traj[:,4],'r')

    plt.show()