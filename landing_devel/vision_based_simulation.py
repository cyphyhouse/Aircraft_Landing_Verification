import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
import rospy
import time
from scipy.integrate import odeint
import rospy 
import rospkg 
import cv2
from cv_bridge import CvBridge, CvBridgeError
import squaternion 
import scipy.spatial 
import pathlib
from PIL import Image as PILImage

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from rosplane_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from sensor_msgs.msg import Image

import torch
import torch.nn.functional as F
from unet import UNet
from utils.data_loading import BasicDataset
import os

from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt 

from aircraft_model import *
from aircraft_mpc import *
from aircraft_simulator import *

import control
from scipy.integrate import odeint

# import aircraft_controller
from agent_aircraft import AircraftTrackingAgent

import copy

import argparse
import logging

script_dir = os.path.dirname(os.path.realpath(__file__)) 
img_path = os.path.join(script_dir, './imgs')

def get_args():
    model = os.path.join(script_dir, 'model.pth')
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=model, metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step, net, device):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point)
        estimate_point = sample_point(estimate_lower, estimate_upper)
        init = np.concatenate((point, estimate_point, ref))
        scenario.set_init(
            [[init]],
            [(FixedWingMode.Normal,)]
        )
        res = scenario.simulate(computation_step, time_step)
        trace = res.nodes[0].trace['a1']
        point = trace[-1,1:7]
        traj.append(np.insert(point, 0, t))
        ref = run_ref(ref, computation_step)
    return traj


if __name__ == "__main__":
    args = get_args()

    rospy.init_node('update_poses', anonymous=True)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


    net = UNet(n_channels=3, n_classes=14, bilinear=args.bilinear) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # Load trained model.
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    fixed_wing_controller = os.path.join(script_dir, './verse/fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    

    try:
        # Run simulation.
        run_vision_sim(net, device)
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
        
