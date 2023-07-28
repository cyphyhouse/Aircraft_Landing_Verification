# Copyright 2023, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model
# using the matrix API:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

K1 = [0.01,0.01,0.01,0.01]
K2 = [0.005,0.005]

try:

    # Create a new model
    m = gp.Model("nonlinear")

    # Create variables
    x = m.addVar(lb=-3050, ub=-3010, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(lb=-20, ub=20, vtype=GRB.CONTINUOUS, name='y')
    z = m.addVar(lb=110, ub=130, vtype=GRB.CONTINUOUS, name='z')
    yaw = m.addVar(lb=-0.001, ub=0.001, vtype=GRB.CONTINUOUS, name='yaw')
    pitch = m.addVar(lb=0.0513598776, ub=0.0533598776, vtype=GRB.CONTINUOUS, name='pitch')
    v = m.addVar(lb=9.99, ub=10.01, vtype=GRB.CONTINUOUS, name='v')
    x_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_next')
    y_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_next')
    z_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_next')
    yaw_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_next')
    pitch_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch`_next')
    v_next = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v_next')
    tmp1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='tmp1')
    tmp2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='tmp2')
    sin_yaw = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='sin_yaw')
    cos_yaw = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='cos_yaw')
    sin_pitch = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='sin_pitch')
    cos_pitch = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='cos_pitch')
    

    x_ref = -3000
    y_ref = 0
    z_ref = 120
    yaw_ref = 0
    pitch_ref = -np.deg2rad(3)
    v_ref = 10
    x_err = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err')
    y_err = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err')
    z_err = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err')
    yaw_err = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err')
    new_v_xy = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy')
    new_v_z = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z')
    new_v = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel')
    yaw_err_cos = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos')
    yaw_err_sin = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin')
    a_input = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input')
    yaw_input = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input')
    pitch_input = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input')
    
    # Set objective
    m.setObjective(
        y_next, GRB.MAXIMIZE
    )

    m.addGenConstrSin(yaw, sin_yaw)
    m.addGenConstrCos(yaw, cos_yaw)
    m.addGenConstrCos(pitch, cos_pitch)
    m.addGenConstrSin(pitch, sin_pitch)
    m.addConstr(x_err==cos_yaw*(x_ref-x)+sin_yaw*(y_ref-y))
    m.addConstr(y_err==-sin_yaw*(x_ref-x)+cos_yaw*(y_ref-y))
    m.addConstr(z_err==z_ref-z)
    m.addConstr(yaw_err==yaw_ref-yaw)
    m.addGenConstrCos(yaw_err, yaw_err_cos)
    m.addGenConstrSin(yaw_err, yaw_err_sin)
    m.addConstr(new_v_xy==v_ref*np.cos(pitch_ref)*yaw_err_cos+K1[0]*x_err)
    m.addConstr(yaw_input==yaw_err+v_ref*(K1[1]*y_err+K1[2]*yaw_err_sin))
    m.addConstr(new_v_z==v_ref*np.sin(pitch_ref)+K1[3]*z_err)
    m.addGenConstrNorm(new_v, [new_v_xy, new_v_z], 2)
    m.addConstr(a_input==K2[0]*(new_v-v))
    m.addConstr(pitch_input==(pitch_ref-pitch)+K2[1]*z_err)
    m.addConstr(tmp1==cos_yaw*cos_pitch)
    m.addConstr(tmp2==sin_yaw*cos_pitch)
    m.addConstr(x_next==x+0.1*v*tmp1)
    m.addConstr(y_next==y+0.1*v*tmp2)
    m.addConstr(z_next==y+0.1*v*sin_pitch)
    m.addConstr(yaw_next==yaw+0.1*yaw_input)
    m.addConstr(pitch_next==pitch+0.1*pitch_input)
    m.addConstr(v_next==v+0.1*a_input)
    # m.computeIIS()

    m.params.NonConvex = 2


    # Optimize model
    m.optimize()

    # print(x.X)
    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')