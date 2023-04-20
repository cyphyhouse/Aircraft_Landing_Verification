# import numpy as np
from casadi import *
import do_mpc

def aircraft_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = 0.1)
    simulator.setup()

    return simulator