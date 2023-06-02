# import numpy as np
from casadi import *
import do_mpc

def aircraft_simulator(model, delta_t):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = delta_t)
    simulator.setup()

    return simulator