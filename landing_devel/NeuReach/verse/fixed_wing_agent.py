from casadi import *
import do_mpc
import numpy as np 
import math
import copy
import matplotlib.pyplot as plt 
from verse.agents import BaseAgent

def aircraft_model(ref_x, scalings=None):
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))

    model.set_rhs('x', vertcat(x[3]*cos(x[4])*cos(x[5]), x[3]*sin(x[4])*cos(x[5]), x[3]*sin(x[5]), u[0], u[1], u[2]))
    # *****************************************************
    # Track pitch.
    # cost = scalings[0]*(x[0] - ref_x[0])**2 + scalings[1]*(x[1] - ref_x[1])**2 + scalings[2]*(x[2] - ref_x[2])**2 + scalings[3]*(x[4] - ref_x[3])**2 
    # cost = scalings[0]*(x[0] - ref_x[0])**2 + scalings[1]*(x[1] - ref_x[1])**2 + scalings[2]*(x[2] - ref_x[2])**2 + scalings[3]*(x[5] - ref_x[3])**2

    cost = (x[0] - ref_x[0])**2 + (x[1] - ref_x[1])**2 + (x[2] - ref_x[2])**2 + (x[5] + np.deg2rad(3))**2 

    term_cost = cost
    # *****************************************************
    model.set_expression('cost', cost)
    model.set_expression('terminal_cost', term_cost)
    model.setup()

    return model

def aircraft_mpc(model, v_max, acc_max, beta_max, omega_max, delta_t):

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 5,
        'n_robust': 0,
        'open_loop': 0,
        't_step': delta_t,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True
    }

    mpc.set_param(**setup_mpc)
  
    mterm = model.aux['terminal_cost']
    lterm = 10.0*model.aux['cost'] 

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=np.array([[0.0],[0.0],[0.0]]))
    # mpc.set_rterm(u=np.array([[0],[0],[0]]))
    max_input = np.array([[acc_max], [beta_max], [omega_max]])
    min_input = np.array([[0.0], [-beta_max], [-omega_max]])
    mpc.bounds['lower', '_u', 'u'] = min_input
    mpc.bounds['upper', '_u', 'u'] = max_input
    x_bounds_max = np.array([[math.inf], [math.inf], [math.inf], [v_max], [15.0*pi/180], [10.0*pi/180]])
    x_bounds_min = np.array([[-math.inf], [-math.inf], [-math.inf], [-v_max], [-15.0*pi/180], [-10.0*pi/180]])
    mpc.bounds['lower', '_x', 'x'] = x_bounds_min
    mpc.bounds['upper', '_x', 'x'] = x_bounds_max

    mpc.setup()

    return mpc

def aircraft_simulator(model, delta_t):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = delta_t)
    simulator.setup()

    return simulator

def run_controller(x_true, x_cur, x_ref, delta_t, v_max=50, acc_max=20, beta_max=0.02, omega_max=0.03):
    '''
    Controller.
    x_true: ground truth of current state.
    x_cur: estimation of current state.
    x_ref: reference waypoint.
    v_max: maximum speed.
    acc_max: maximum acceleration.
    beta_max: maximum yaw rate.
    omage_max: maximum pitch rate.
    '''
    model = aircraft_model(x_ref)
    mpc = aircraft_mpc(model, v_max, acc_max, beta_max, omega_max, delta_t)
    simulator = aircraft_simulator(model, delta_t)
    simulator.x0 = np.array(x_true)
    mpc.x0 = x_cur

    u_init = np.full((3, 1), 0.0)
    mpc.u0 = u_init
    simulator.u0 = u_init
    mpc.set_initial_guess()

    u0 = mpc.make_step(x_cur)
    x_next = simulator.make_step(u0)

    return x_next

def run_ref(ref_state, time_step, approaching_angle):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = 50*time_step
    delta_z = k*delta_x*time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z])

def TC_simulate(mode, init, time_bound, time_step, lane_map):
    time_steps = np.arange(0,time_bound, time_step)
    
    state = np.array(init)
    trajectory = copy.deepcopy(state)
    trajectory = np.insert(trajectory, 0, time_steps[0])
    trajectory = np.reshape(trajectory,(1,-1))
    for i in range(1,len(time_steps)):
        x_ground_truth = state[:6]
        ref_state = state[6:]
        x_next = run_controller(x_ground_truth, x_ground_truth, ref_state, time_step).squeeze()
        ref_next = run_ref(ref_state, time_step, approaching_angle = 3)
        state = np.concatenate((x_next, ref_next)) 
        tmp = np.insert(state, 0, time_steps[i])
        tmp = np.reshape(tmp,(1,-1))
        trajectory = np.vstack((trajectory, tmp))

    return trajectory

class FixedWingAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)

    def TC_simulate(self, mode, init, time_bound, time_step, lane_map):
        time_steps = np.arange(0,time_bound, time_step)
        
        state = np.array(init)
        trajectory = copy.deepcopy(state)
        trajectory = np.insert(trajectory, 0, time_steps[0])
        trajectory = np.reshape(trajectory,(1,-1))
        for i in range(1,len(time_steps)):
            x_ground_truth = state[:6]
            ref_state = state[6:]
            x_next = run_controller(x_ground_truth, x_ground_truth, ref_state, time_step).squeeze()
            x_next[3] = 0
            ref_next = run_ref(ref_state, time_step, approaching_angle = 3)
            state = np.concatenate((x_next, ref_next)) 
            tmp = np.insert(state, 0, time_steps[i])
            tmp = np.reshape(tmp,(1,-1))
            trajectory = np.vstack((trajectory, tmp))

        return trajectory

if __name__ == "__main__":
    agent = FixedWingAgent()
    init = np.array([-2530.0, 10, 120.0, 0, -np.deg2rad(3), 0, -2500.0, 0, 120.0])
    traj = agent.TC_simulate(None, init, 15, 0.05, None)
    print(traj)

    plt.figure()
    plt.plot(traj[:,1], traj[:,2])

    plt.figure()
    plt.plot(traj[:,1], traj[:,3])
    plt.show()