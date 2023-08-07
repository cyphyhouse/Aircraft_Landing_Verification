import numpy as np 
import copy
from enum import Enum, auto

class ObjectiveType(Enum):
    MIN = auto()
    MAX = auto()

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])


if __name__ == "__main__":
    x_init = [-3050, -3010]
    y_init = [-20, 20]
    z_init = [110, 130]
    yaw_init = [-0.001, 0.001]
    pitch_init = [-np.deg2rad(3)-0.001, -np.deg2rad(3)+0.001]
    v_init = [9.99, 10.01]
    ref0 = np.array([-3000, 0, 120, 0, -np.deg2rad(3), 10])
    K1 = [0.01,0.01,0.01,0.01]
    K2 = [0.005,0.005]

    C_compute_step = 7
    computation_step = 0.1

    objective_variable = 'yaw'
    objective_type = ObjectiveType.MAX

    if objective_type == ObjectiveType.MAX:
        objective_str = 'GRB.MAXIMIZE'
    elif objective_type == ObjectiveType.MIN:
        objective_str = 'GRB.MINIMIZE'    

    model_header_str = "\
import gurobipy as gp\n\
from gurobipy import GRB\n\
import numpy as np\n\
import scipy.sparse as sp\n\
\n\
try:\n\
    m = gp.Model('nonlinear')\n"
    model_init_str = f"\
    x0 = m.addVar(lb={x_init[0]}, ub={x_init[1]}, vtype=GRB.CONTINUOUS, name='x0')\n\
    y0 = m.addVar(lb={y_init[0]}, ub={y_init[1]}, vtype=GRB.CONTINUOUS, name='y0')\n\
    z0 = m.addVar(lb={z_init[0]}, ub={z_init[1]}, vtype=GRB.CONTINUOUS, name='z0')\n\
    yaw0 = m.addVar(lb={yaw_init[0]}, ub={yaw_init[1]}, vtype=GRB.CONTINUOUS, name='yaw0')\n\
    pitch0 = m.addVar(lb={pitch_init[0]}, ub={pitch_init[1]}, vtype=GRB.CONTINUOUS, name='pitch0')\n\
    v0 = m.addVar(lb={v_init[0]}, ub={v_init[1]}, vtype=GRB.CONTINUOUS, name='v0')\n\
\n"

    model_step_str_list = []
    ref = copy.deepcopy(ref0)
    for i in range(C_compute_step):
        model_step_var_str = f"\
    # Decalre Variables for step {i}\n\
    cos_yaw_cos_pitch{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw_cos_pitch{i}')\n\
    sin_yaw_cos_pitch{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw_cos_pitch{i}')\n\
    sin_yaw{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_yaw{i}')\n\
    cos_yaw{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_yaw{i}')\n\
    sin_pitch{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='sin_pitch{i}')\n\
    cos_pitch{i} = m.addVar(lb=-1.1, ub=1.1, vtype=GRB.CONTINUOUS, name='cos_pitch{i}')\n\
\n\
    x_err{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_err{i}')\n\
    y_err{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_err{i}')\n\
    z_err{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z_err{i}')\n\
    yaw_err{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err{i}')\n\
    new_v_xy{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_xy{i}')\n\
    new_v_z{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_vel_z{i}')\n\
    new_v{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='new_v{i}')\n\
    yaw_err_cos{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_cos{i}')\n\
    yaw_err_sin{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_err_sin{i}')\n\
    a_input{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='a_input{i}')\n\
    yaw_input{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw_input{i}')\n\
    pitch_input{i} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch_input{i}')\n\
\n\
    x{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x{i+1}')\n\
    y{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y{i+1}')\n\
    z{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='z{i+1}')\n\
    yaw{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='yaw{i+1}')\n\
    pitch{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='pitch{i+1}')\n\
    v{i+1} = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='v{i+1}')\n\
\n"

        model_step_dynamics_str = f"\
    # Dynamics for step {i}\n\
    m.addGenConstrSin(yaw{i}, sin_yaw{i})\n\
    m.addGenConstrCos(yaw{i}, cos_yaw{i})\n\
    m.addGenConstrCos(pitch{i}, cos_pitch{i})\n\
    m.addGenConstrSin(pitch{i}, sin_pitch{i})\n\
    m.addConstr(x_err{i}==cos_yaw{i}*({ref[0]}-x{i})+sin_yaw{i}*({ref[1]}-y{i}))\n\
    m.addConstr(y_err{i}==-sin_yaw{i}*({ref[0]}-x{i})+cos_yaw{i}*({ref[1]}-y{i}))\n\
    m.addConstr(z_err{i}=={ref[2]}-z{i})\n\
    m.addConstr(yaw_err{i}=={ref[3]}-yaw{i})\n\
    m.addGenConstrCos(yaw_err{i}, yaw_err_cos{i})\n\
    m.addGenConstrSin(yaw_err{i}, yaw_err_sin{i})\n\
    m.addConstr(new_v_xy{i}=={ref[5]}*({np.cos(ref[4])})*yaw_err_cos{i}+{K1[0]}*x_err{i})\n\
    m.addConstr(yaw_input{i}==yaw_err{i}+{ref[5]}*({K1[1]}*y_err{i}+{K1[2]}*yaw_err_sin{i}))\n\
    m.addConstr(new_v_z{i}=={ref[5]}*{np.sin(ref[4])}+{K1[3]}*z_err{i})\n\
    m.addGenConstrNorm(new_v{i}, [new_v_xy{i}, new_v_z{i}], 2)\n\
    m.addConstr(a_input{i}=={K2[0]}*(new_v{i}-v{i}))\n\
    m.addConstr(pitch_input{i}==({ref[4]}-pitch{i})+{K2[1]}*z_err{i})\n\
    m.addConstr(cos_yaw_cos_pitch{i}==cos_yaw{i}*cos_pitch{i})\n\
    m.addConstr(sin_yaw_cos_pitch{i}==sin_yaw{i}*cos_pitch{i})\n\
    m.addConstr(x{i+1}==x{i}+{computation_step}*v{i}*cos_yaw_cos_pitch{i})\n\
    m.addConstr(y{i+1}==y{i}+{computation_step}*v{i}*sin_yaw_cos_pitch{i})\n\
    m.addConstr(z{i+1}==z{i}+{computation_step}*v{i}*sin_pitch{i})\n\
    m.addConstr(yaw{i+1}==yaw{i}+{computation_step}*yaw_input{i})\n\
    m.addConstr(pitch{i+1}==pitch{i}+{computation_step}*pitch_input{i})\n\
    m.addConstr(v{i+1}==v{i}+{computation_step}*a_input{i})\n\
\n\
"
        ref = run_ref(ref, computation_step)
        model_step_str = model_step_var_str + model_step_dynamics_str
        model_step_str_list.append(model_step_str)

    model_end_str = f"\
    # Set objective\n\
    m.setObjective(\n\
        {objective_variable}{C_compute_step}, {objective_str}\n\
    )\n\
\n\
    m.params.NonConvex = 2\n\
\n\
\n\
    # Optimize model\n\
    m.optimize()\n\
\n\
    # print(x.X)\n\
    print('Obj: %g' % m.ObjVal)\n\
\n\
    # m.write('model.lp')\n\
\n\
except gp.GurobiError as e:\n\
    print('Error code ' + str(e.errno) + ': ' + str(e))\n\
\n\
except AttributeError:\n\
    print('Encountered an attribute error')\n\
    "

    model_str = model_header_str + model_init_str
    for model_step_str in model_step_str_list:
        model_str = model_str + model_step_str 
    
    model_str = model_str + model_end_str

    with open('gurobi_model_ics.py','w+') as f:
        f.write(model_str)
    
    # y = 19.3362