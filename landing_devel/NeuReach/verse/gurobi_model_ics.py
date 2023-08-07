import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

try:
    m = gp.Model('nonlinear')
    x0 = m.addVar(lb=-3050, ub=-3010, vtype=GRB.CONTINUOUS, name='x0')
    y0 = m.addVar(lb=-20, ub=20, vtype=GRB.CONTINUOUS, name='y0')
    z0 = m.addVar(lb=110, ub=130, vtype=GRB.CONTINUOUS, name='z0')
    yaw0 = m.addVar(lb=-0.001, ub=0.001, vtype=GRB.CONTINUOUS, name='yaw0')
    pitch0 = m.addVar(lb=-0.05335987755982989, ub=-0.05135987755982989, vtype=GRB.CONTINUOUS, name='pitch0')
    v0 = m.addVar(lb=9.99, ub=10.01, vtype=GRB.CONTINUOUS, name='v0')

    # Decalre Variables for step 0
    cos_yaw_cos_pitch0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch0')
    sin_yaw_cos_pitch0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch0')
    sin_yaw0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw0')
    cos_yaw0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw0')
    sin_pitch0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch0')
    cos_pitch0 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch0')

    x_err0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err0')
    y_err0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err0')
    z_err0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err0')
    yaw_err0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err0')
    new_v_xy0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy0')
    new_v_z0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z0')
    new_v0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v0')
    yaw_err_cos0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos0')
    yaw_err_sin0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin0')
    a_input0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input0')
    yaw_input0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input0')
    pitch_input0 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input0')

    x1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x1')
    y1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y1')
    z1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z1')
    yaw1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw1')
    pitch1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch1')
    v1 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v1')

    # Dynamics for step 0
    m.addGenConstrSin(yaw0, sin_yaw0)
    m.addGenConstrCos(yaw0, cos_yaw0)
    m.addGenConstrCos(pitch0, cos_pitch0)
    m.addGenConstrSin(pitch0, sin_pitch0)
    m.addConstr(x_err0==cos_yaw0*(-3000.0-x0)+sin_yaw0*(0.0-y0))
    m.addConstr(y_err0==-sin_yaw0*(-3000.0-x0)+cos_yaw0*(0.0-y0))
    m.addConstr(z_err0==120.0-z0)
    m.addConstr(yaw_err0==0.0-yaw0)
    m.addGenConstrCos(yaw_err0, yaw_err_cos0)
    m.addGenConstrSin(yaw_err0, yaw_err_sin0)
    m.addConstr(new_v_xy0==10.0*(0.9986295347545738)*yaw_err_cos0+0.01*x_err0)
    m.addConstr(yaw_input0==yaw_err0+10.0*(0.01*y_err0+0.01*yaw_err_sin0))
    m.addConstr(new_v_z0==10.0*-0.052335956242943835+0.01*z_err0)
    m.addGenConstrNorm(new_v0, [new_v_xy0, new_v_z0], 2)
    m.addConstr(a_input0==0.005*(new_v0-v0))
    m.addConstr(pitch_input0==(-0.05235987755982989-pitch0)+0.005*z_err0)
    m.addConstr(cos_yaw_cos_pitch0==cos_yaw0*cos_pitch0)
    m.addConstr(sin_yaw_cos_pitch0==sin_yaw0*cos_pitch0)
    m.addConstr(x1==x0+0.1*v0*cos_yaw_cos_pitch0)
    m.addConstr(y1==y0+0.1*v0*sin_yaw_cos_pitch0)
    m.addConstr(z1==z0+0.1*v0*sin_pitch0)
    m.addConstr(yaw1==yaw0+0.1*yaw_input0)
    m.addConstr(pitch1==pitch0+0.1*pitch_input0)
    m.addConstr(v1==v0+0.1*a_input0)

    # Decalre Variables for step 1
    cos_yaw_cos_pitch1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch1')
    sin_yaw_cos_pitch1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch1')
    sin_yaw1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw1')
    cos_yaw1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw1')
    sin_pitch1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch1')
    cos_pitch1 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch1')

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

    x2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x2')
    y2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y2')
    z2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z2')
    yaw2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw2')
    pitch2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch2')
    v2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v2')

    # Dynamics for step 1
    m.addGenConstrSin(yaw1, sin_yaw1)
    m.addGenConstrCos(yaw1, cos_yaw1)
    m.addGenConstrCos(pitch1, cos_pitch1)
    m.addGenConstrSin(pitch1, sin_pitch1)
    m.addConstr(x_err1==cos_yaw1*(-2999.0-x1)+sin_yaw1*(0.0-y1))
    m.addConstr(y_err1==-sin_yaw1*(-2999.0-x1)+cos_yaw1*(0.0-y1))
    m.addConstr(z_err1==119.94759222071696-z1)
    m.addConstr(yaw_err1==0.0-yaw1)
    m.addGenConstrCos(yaw_err1, yaw_err_cos1)
    m.addGenConstrSin(yaw_err1, yaw_err_sin1)
    m.addConstr(new_v_xy1==10.0*(0.9986295347545738)*yaw_err_cos1+0.01*x_err1)
    m.addConstr(yaw_input1==yaw_err1+10.0*(0.01*y_err1+0.01*yaw_err_sin1))
    m.addConstr(new_v_z1==10.0*-0.052335956242943835+0.01*z_err1)
    m.addGenConstrNorm(new_v1, [new_v_xy1, new_v_z1], 2)
    m.addConstr(a_input1==0.005*(new_v1-v1))
    m.addConstr(pitch_input1==(-0.05235987755982989-pitch1)+0.005*z_err1)
    m.addConstr(cos_yaw_cos_pitch1==cos_yaw1*cos_pitch1)
    m.addConstr(sin_yaw_cos_pitch1==sin_yaw1*cos_pitch1)
    m.addConstr(x2==x1+0.1*v1*cos_yaw_cos_pitch1)
    m.addConstr(y2==y1+0.1*v1*sin_yaw_cos_pitch1)
    m.addConstr(z2==z1+0.1*v1*sin_pitch1)
    m.addConstr(yaw2==yaw1+0.1*yaw_input1)
    m.addConstr(pitch2==pitch1+0.1*pitch_input1)
    m.addConstr(v2==v1+0.1*a_input1)

    # Decalre Variables for step 2
    cos_yaw_cos_pitch2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch2')
    sin_yaw_cos_pitch2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch2')
    sin_yaw2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw2')
    cos_yaw2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw2')
    sin_pitch2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch2')
    cos_pitch2 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch2')

    x_err2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err2')
    y_err2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err2')
    z_err2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err2')
    yaw_err2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err2')
    new_v_xy2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy2')
    new_v_z2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z2')
    new_v2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v2')
    yaw_err_cos2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos2')
    yaw_err_sin2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin2')
    a_input2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input2')
    yaw_input2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input2')
    pitch_input2 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input2')

    x3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x3')
    y3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y3')
    z3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z3')
    yaw3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw3')
    pitch3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch3')
    v3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v3')

    # Dynamics for step 2
    m.addGenConstrSin(yaw2, sin_yaw2)
    m.addGenConstrCos(yaw2, cos_yaw2)
    m.addGenConstrCos(pitch2, cos_pitch2)
    m.addGenConstrSin(pitch2, sin_pitch2)
    m.addConstr(x_err2==cos_yaw2*(-2998.0-x2)+sin_yaw2*(0.0-y2))
    m.addConstr(y_err2==-sin_yaw2*(-2998.0-x2)+cos_yaw2*(0.0-y2))
    m.addConstr(z_err2==119.89518444143391-z2)
    m.addConstr(yaw_err2==0.0-yaw2)
    m.addGenConstrCos(yaw_err2, yaw_err_cos2)
    m.addGenConstrSin(yaw_err2, yaw_err_sin2)
    m.addConstr(new_v_xy2==10.0*(0.9986295347545738)*yaw_err_cos2+0.01*x_err2)
    m.addConstr(yaw_input2==yaw_err2+10.0*(0.01*y_err2+0.01*yaw_err_sin2))
    m.addConstr(new_v_z2==10.0*-0.052335956242943835+0.01*z_err2)
    m.addGenConstrNorm(new_v2, [new_v_xy2, new_v_z2], 2)
    m.addConstr(a_input2==0.005*(new_v2-v2))
    m.addConstr(pitch_input2==(-0.05235987755982989-pitch2)+0.005*z_err2)
    m.addConstr(cos_yaw_cos_pitch2==cos_yaw2*cos_pitch2)
    m.addConstr(sin_yaw_cos_pitch2==sin_yaw2*cos_pitch2)
    m.addConstr(x3==x2+0.1*v2*cos_yaw_cos_pitch2)
    m.addConstr(y3==y2+0.1*v2*sin_yaw_cos_pitch2)
    m.addConstr(z3==z2+0.1*v2*sin_pitch2)
    m.addConstr(yaw3==yaw2+0.1*yaw_input2)
    m.addConstr(pitch3==pitch2+0.1*pitch_input2)
    m.addConstr(v3==v2+0.1*a_input2)

    # Decalre Variables for step 3
    cos_yaw_cos_pitch3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch3')
    sin_yaw_cos_pitch3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch3')
    sin_yaw3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw3')
    cos_yaw3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw3')
    sin_pitch3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch3')
    cos_pitch3 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch3')

    x_err3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err3')
    y_err3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err3')
    z_err3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err3')
    yaw_err3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err3')
    new_v_xy3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy3')
    new_v_z3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z3')
    new_v3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v3')
    yaw_err_cos3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos3')
    yaw_err_sin3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin3')
    a_input3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input3')
    yaw_input3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input3')
    pitch_input3 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input3')

    x4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x4')
    y4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y4')
    z4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z4')
    yaw4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw4')
    pitch4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch4')
    v4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v4')

    # Dynamics for step 3
    m.addGenConstrSin(yaw3, sin_yaw3)
    m.addGenConstrCos(yaw3, cos_yaw3)
    m.addGenConstrCos(pitch3, cos_pitch3)
    m.addGenConstrSin(pitch3, sin_pitch3)
    m.addConstr(x_err3==cos_yaw3*(-2997.0-x3)+sin_yaw3*(0.0-y3))
    m.addConstr(y_err3==-sin_yaw3*(-2997.0-x3)+cos_yaw3*(0.0-y3))
    m.addConstr(z_err3==119.84277666215087-z3)
    m.addConstr(yaw_err3==0.0-yaw3)
    m.addGenConstrCos(yaw_err3, yaw_err_cos3)
    m.addGenConstrSin(yaw_err3, yaw_err_sin3)
    m.addConstr(new_v_xy3==10.0*(0.9986295347545738)*yaw_err_cos3+0.01*x_err3)
    m.addConstr(yaw_input3==yaw_err3+10.0*(0.01*y_err3+0.01*yaw_err_sin3))
    m.addConstr(new_v_z3==10.0*-0.052335956242943835+0.01*z_err3)
    m.addGenConstrNorm(new_v3, [new_v_xy3, new_v_z3], 2)
    m.addConstr(a_input3==0.005*(new_v3-v3))
    m.addConstr(pitch_input3==(-0.05235987755982989-pitch3)+0.005*z_err3)
    m.addConstr(cos_yaw_cos_pitch3==cos_yaw3*cos_pitch3)
    m.addConstr(sin_yaw_cos_pitch3==sin_yaw3*cos_pitch3)
    m.addConstr(x4==x3+0.1*v3*cos_yaw_cos_pitch3)
    m.addConstr(y4==y3+0.1*v3*sin_yaw_cos_pitch3)
    m.addConstr(z4==z3+0.1*v3*sin_pitch3)
    m.addConstr(yaw4==yaw3+0.1*yaw_input3)
    m.addConstr(pitch4==pitch3+0.1*pitch_input3)
    m.addConstr(v4==v3+0.1*a_input3)

    # Decalre Variables for step 4
    cos_yaw_cos_pitch4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch4')
    sin_yaw_cos_pitch4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch4')
    sin_yaw4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw4')
    cos_yaw4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw4')
    sin_pitch4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch4')
    cos_pitch4 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch4')

    x_err4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err4')
    y_err4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err4')
    z_err4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err4')
    yaw_err4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err4')
    new_v_xy4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy4')
    new_v_z4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z4')
    new_v4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v4')
    yaw_err_cos4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos4')
    yaw_err_sin4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin4')
    a_input4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input4')
    yaw_input4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input4')
    pitch_input4 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input4')

    x5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x5')
    y5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y5')
    z5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z5')
    yaw5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw5')
    pitch5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch5')
    v5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v5')

    # Dynamics for step 4
    m.addGenConstrSin(yaw4, sin_yaw4)
    m.addGenConstrCos(yaw4, cos_yaw4)
    m.addGenConstrCos(pitch4, cos_pitch4)
    m.addGenConstrSin(pitch4, sin_pitch4)
    m.addConstr(x_err4==cos_yaw4*(-2996.0-x4)+sin_yaw4*(0.0-y4))
    m.addConstr(y_err4==-sin_yaw4*(-2996.0-x4)+cos_yaw4*(0.0-y4))
    m.addConstr(z_err4==119.79036888286782-z4)
    m.addConstr(yaw_err4==0.0-yaw4)
    m.addGenConstrCos(yaw_err4, yaw_err_cos4)
    m.addGenConstrSin(yaw_err4, yaw_err_sin4)
    m.addConstr(new_v_xy4==10.0*(0.9986295347545738)*yaw_err_cos4+0.01*x_err4)
    m.addConstr(yaw_input4==yaw_err4+10.0*(0.01*y_err4+0.01*yaw_err_sin4))
    m.addConstr(new_v_z4==10.0*-0.052335956242943835+0.01*z_err4)
    m.addGenConstrNorm(new_v4, [new_v_xy4, new_v_z4], 2)
    m.addConstr(a_input4==0.005*(new_v4-v4))
    m.addConstr(pitch_input4==(-0.05235987755982989-pitch4)+0.005*z_err4)
    m.addConstr(cos_yaw_cos_pitch4==cos_yaw4*cos_pitch4)
    m.addConstr(sin_yaw_cos_pitch4==sin_yaw4*cos_pitch4)
    m.addConstr(x5==x4+0.1*v4*cos_yaw_cos_pitch4)
    m.addConstr(y5==y4+0.1*v4*sin_yaw_cos_pitch4)
    m.addConstr(z5==z4+0.1*v4*sin_pitch4)
    m.addConstr(yaw5==yaw4+0.1*yaw_input4)
    m.addConstr(pitch5==pitch4+0.1*pitch_input4)
    m.addConstr(v5==v4+0.1*a_input4)

    # Decalre Variables for step 5
    cos_yaw_cos_pitch5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch5')
    sin_yaw_cos_pitch5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch5')
    sin_yaw5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw5')
    cos_yaw5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw5')
    sin_pitch5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch5')
    cos_pitch5 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch5')

    x_err5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err5')
    y_err5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err5')
    z_err5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err5')
    yaw_err5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err5')
    new_v_xy5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy5')
    new_v_z5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z5')
    new_v5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v5')
    yaw_err_cos5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos5')
    yaw_err_sin5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin5')
    a_input5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input5')
    yaw_input5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input5')
    pitch_input5 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input5')

    x6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x6')
    y6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y6')
    z6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z6')
    yaw6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw6')
    pitch6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch6')
    v6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v6')

    # Dynamics for step 5
    m.addGenConstrSin(yaw5, sin_yaw5)
    m.addGenConstrCos(yaw5, cos_yaw5)
    m.addGenConstrCos(pitch5, cos_pitch5)
    m.addGenConstrSin(pitch5, sin_pitch5)
    m.addConstr(x_err5==cos_yaw5*(-2995.0-x5)+sin_yaw5*(0.0-y5))
    m.addConstr(y_err5==-sin_yaw5*(-2995.0-x5)+cos_yaw5*(0.0-y5))
    m.addConstr(z_err5==119.73796110358478-z5)
    m.addConstr(yaw_err5==0.0-yaw5)
    m.addGenConstrCos(yaw_err5, yaw_err_cos5)
    m.addGenConstrSin(yaw_err5, yaw_err_sin5)
    m.addConstr(new_v_xy5==10.0*(0.9986295347545738)*yaw_err_cos5+0.01*x_err5)
    m.addConstr(yaw_input5==yaw_err5+10.0*(0.01*y_err5+0.01*yaw_err_sin5))
    m.addConstr(new_v_z5==10.0*-0.052335956242943835+0.01*z_err5)
    m.addGenConstrNorm(new_v5, [new_v_xy5, new_v_z5], 2)
    m.addConstr(a_input5==0.005*(new_v5-v5))
    m.addConstr(pitch_input5==(-0.05235987755982989-pitch5)+0.005*z_err5)
    m.addConstr(cos_yaw_cos_pitch5==cos_yaw5*cos_pitch5)
    m.addConstr(sin_yaw_cos_pitch5==sin_yaw5*cos_pitch5)
    m.addConstr(x6==x5+0.1*v5*cos_yaw_cos_pitch5)
    m.addConstr(y6==y5+0.1*v5*sin_yaw_cos_pitch5)
    m.addConstr(z6==z5+0.1*v5*sin_pitch5)
    m.addConstr(yaw6==yaw5+0.1*yaw_input5)
    m.addConstr(pitch6==pitch5+0.1*pitch_input5)
    m.addConstr(v6==v5+0.1*a_input5)

    # Decalre Variables for step 6
    cos_yaw_cos_pitch6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch6')
    sin_yaw_cos_pitch6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch6')
    sin_yaw6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw6')
    cos_yaw6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw6')
    sin_pitch6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch6')
    cos_pitch6 = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch6')

    x_err6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err6')
    y_err6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err6')
    z_err6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err6')
    yaw_err6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err6')
    new_v_xy6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy6')
    new_v_z6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z6')
    new_v6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v6')
    yaw_err_cos6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos6')
    yaw_err_sin6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin6')
    a_input6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input6')
    yaw_input6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input6')
    pitch_input6 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input6')

    x7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x7')
    y7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y7')
    z7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z7')
    yaw7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw7')
    pitch7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch7')
    v7 = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v7')

    # Dynamics for step 6
    m.addGenConstrSin(yaw6, sin_yaw6)
    m.addGenConstrCos(yaw6, cos_yaw6)
    m.addGenConstrCos(pitch6, cos_pitch6)
    m.addGenConstrSin(pitch6, sin_pitch6)
    m.addConstr(x_err6==cos_yaw6*(-2994.0-x6)+sin_yaw6*(0.0-y6))
    m.addConstr(y_err6==-sin_yaw6*(-2994.0-x6)+cos_yaw6*(0.0-y6))
    m.addConstr(z_err6==119.68555332430174-z6)
    m.addConstr(yaw_err6==0.0-yaw6)
    m.addGenConstrCos(yaw_err6, yaw_err_cos6)
    m.addGenConstrSin(yaw_err6, yaw_err_sin6)
    m.addConstr(new_v_xy6==10.0*(0.9986295347545738)*yaw_err_cos6+0.01*x_err6)
    m.addConstr(yaw_input6==yaw_err6+10.0*(0.01*y_err6+0.01*yaw_err_sin6))
    m.addConstr(new_v_z6==10.0*-0.052335956242943835+0.01*z_err6)
    m.addGenConstrNorm(new_v6, [new_v_xy6, new_v_z6], 2)
    m.addConstr(a_input6==0.005*(new_v6-v6))
    m.addConstr(pitch_input6==(-0.05235987755982989-pitch6)+0.005*z_err6)
    m.addConstr(cos_yaw_cos_pitch6==cos_yaw6*cos_pitch6)
    m.addConstr(sin_yaw_cos_pitch6==sin_yaw6*cos_pitch6)
    m.addConstr(x7==x6+0.1*v6*cos_yaw_cos_pitch6)
    m.addConstr(y7==y6+0.1*v6*sin_yaw_cos_pitch6)
    m.addConstr(z7==z6+0.1*v6*sin_pitch6)
    m.addConstr(yaw7==yaw6+0.1*yaw_input6)
    m.addConstr(pitch7==pitch6+0.1*pitch_input6)
    m.addConstr(v7==v6+0.1*a_input6)

    # Set objective
    m.setObjective(
        y7, GRB.MAXIMIZE
    )

    m.params.NonConvex = 2


    # Optimize model
    m.optimize()

    # print(x.X)
    print('Obj: %g' % m.ObjVal)

    # m.write('model.lp')

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
    