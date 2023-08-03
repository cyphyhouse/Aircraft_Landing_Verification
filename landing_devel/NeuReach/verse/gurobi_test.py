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
    x1 = m.addVar(lb=-3050, ub=-3010, vtype=GRB.CONTINUOUS, name="x1")
    y1 = m.addVar(lb=-20, ub=20, vtype=GRB.CONTINUOUS, name='y1')
    z1 = m.addVar(lb=110, ub=130, vtype=GRB.CONTINUOUS, name='z1')
    yaw1 = m.addVar(lb=-0.001, ub=0.001, vtype=GRB.CONTINUOUS, name='yaw1')
    pitch1 = m.addVar(lb=0.0513598776, ub=0.0533598776, vtype=GRB.CONTINUOUS, name='pitch1')
    v1 = m.addVar(lb=9.99, ub=10.01, vtype=GRB.CONTINUOUS, name='v1')
    x2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x2')
    y2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y2')
    z2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z2')
    yaw2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw2')
    pitch2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch2')
    v2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v2')
    cos_yaw_cos_pitch1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch1')
    sin_yaw_cos_pitch1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch1')
    sin_yaw1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='sin_yaw1')
    cos_yaw1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='cos_yaw1')
    sin_pitch1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='sin_pitch1')
    cos_pitch1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='cos_pitch1')
    

    x_ref1 = -3000
    y_ref1 = 0
    z_ref1 = 120
    yaw_ref1 = 0
    pitch_ref1 = -np.deg2rad(3)
    v_ref1 = 10
    x_err1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err1')
    y_err1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err1')
    z_err1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err1')
    yaw_err1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err1')
    new_v_xy1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy1')
    new_v_z1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z1')
    new_v1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v1')
    yaw_err_cos1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos1')
    yaw_err_sin1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin1')
    a_input1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input1')
    yaw_input1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input1')
    pitch_input1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input1')
    
    # Set objective
    m.setObjective(
        y2, GRB.MAXIMIZE
    )

    m.addGenConstrSin(yaw1, sin_yaw1)
    m.addGenConstrCos(yaw1, cos_yaw1)
    m.addGenConstrCos(pitch1, cos_pitch1)
    m.addGenConstrSin(pitch1, sin_pitch1)
    m.addConstr(x_err1==cos_yaw1*(x_ref1-x1)+sin_yaw1*(y_ref1-y1))
    m.addConstr(y_err1==-sin_yaw1*(x_ref1-x1)+cos_yaw1*(y_ref1-y1))
    m.addConstr(z_err1==z_ref1-z1)
    m.addConstr(yaw_err1==yaw_ref1-yaw1)
    m.addGenConstrCos(yaw_err1, yaw_err_cos1)
    m.addGenConstrSin(yaw_err1, yaw_err_sin1)
    m.addConstr(new_v_xy1==v_ref1*np.cos(pitch_ref1)*yaw_err_cos1+K1[0]*x_err1)
    m.addConstr(yaw_input1==yaw_err1+v_ref1*(K1[1]*y_err1+K1[2]*yaw_err_sin1))
    m.addConstr(new_v_z1==v_ref1*np.sin(pitch_ref1)+K1[3]*z_err1)
    m.addGenConstrNorm(new_v1, [new_v_xy1, new_v_z1], 2)
    m.addConstr(a_input1==K2[0]*(new_v1-v1))
    m.addConstr(pitch_input1==(pitch_ref1-pitch1)+K2[1]*z_err1)
    m.addConstr(cos_yaw_cos_pitch1==cos_yaw1*cos_pitch1)
    m.addConstr(sin_yaw_cos_pitch1==sin_yaw1*cos_pitch1)
    m.addConstr(x2==x1+0.1*v1*cos_yaw_cos_pitch1)
    m.addConstr(y2==y1+0.1*v1*sin_yaw_cos_pitch1)
    m.addConstr(z2==y1+0.1*v1*sin_pitch1)
    m.addConstr(yaw2==yaw1+0.1*yaw_input1)
    m.addConstr(pitch2==pitch1+0.1*pitch_input1)
    m.addConstr(v2==v1+0.1*a_input1)
 
    m.params.NonConvex = 2


    # Optimize model
    m.optimize()

    # print(x.X)
    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')