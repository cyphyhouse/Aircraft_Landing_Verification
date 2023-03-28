'''
Model of the aircraft.
'''
# import numpy as np
from casadi import *
import do_mpc

def aircraft_model(ref_x):
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))

    model.set_rhs('x', vertcat(x[3]*cos(x[4])*cos(x[5]), x[3]*sin(x[4])*cos(x[5]), x[3]*sin(x[5]), u[0], u[1], u[2]))
    # *****************************************************
    cost = (x[0] - ref_x[0])**2 + (x[1] - ref_x[1])**2 + (x[2] - ref_x[2])**2 + (x[4] - ref_x[4])**2 + (x[5] - ref_x[5])**2
    term_cost = cost
    # *****************************************************
    model.set_expression('cost', cost)
    model.set_expression('terminal_cost', term_cost)
    model.setup()

    return model

