import numpy as np
from casadi import *
import do_mpc
import math
def aircraft_mpc(model, v_max, acc_max, beta_max, omega_max):

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 5,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.1,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True
    }

    mpc.set_param(**setup_mpc)
  
    mterm = 10*model.aux['terminal_cost']
    lterm = 10*model.aux['cost'] 

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=np.array([[0],[100],[100]]))
    # mpc.set_rterm(u=np.array([[0],[0],[0]]))
    max_input = np.array([[acc_max], [beta_max], [omega_max]])
    mpc.bounds['lower', '_u', 'u'] = -max_input
    mpc.bounds['upper', '_u', 'u'] = max_input
    x_bounds_max = np.array([[math.inf], [math.inf], [math.inf], [v_max], [3*pi/180], [pi/180]])
    x_bounds_min = np.array([[-math.inf], [-math.inf], [-math.inf], [-v_max], [-3*pi/180], [-4*pi/180]])
    mpc.bounds['lower', '_x', 'x'] = x_bounds_min
    mpc.bounds['upper', '_x', 'x'] = x_bounds_max

    mpc.setup()

    return mpc