import sys
sys.path.append('..')
from ODEs.vanderpol import TC_Simulate
import numpy as np
import os


data_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data.txt'
label_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label.txt'
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')
data_train = data[:, 1:]
label_train = label[:, 1:]
def sample_x0():
    return data[np.random.choice(len(data), size=1, replace=False)]


def simulate(x0):
    return np.array(TC_Simulate("Default", x0, TMAX))

# def get_init_center(x0):
#     return x0
