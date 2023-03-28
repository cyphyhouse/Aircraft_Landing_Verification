import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
import rospy
# import sys
# from typing import List
import time
from scipy.integrate import odeint
import rospy 
import rospkg 
import cv2
from cv_bridge import CvBridge, CvBridgeError
import squaternion 
import scipy.spatial 
import pathlib
# import PIL 
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
# img_path = '/home/younger/work/Aircraft_landing_verification/src/landing_devel/imgs'
img_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/imgs'
class Perception():
    def __init__(self, net, device, name=''):
        self.name = name

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()

        self.count = -1
        self.pic_index = 0

        # Camera matrix
        self.K = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])

        self.net = net
        self.device = device
        self.keypoints = [[-1221.370483, 16.052534, 0.0],
                        [-1279.224854, 16.947235, 0.0],
                        [-1279.349731, 8.911615, 0.0],
                        [-1221.505737, 8.033512, 0.0],
                        [-1221.438110, -8.496282, 0.0],
                        [-1279.302002, -8.493725, 0.0],
                        [-1279.315796, -16.504263, 0.0],
                        [-1221.462402, -16.498976, 0.0],
                        [-1520.81, 26.125700, 0.0],
                        [-1559.122925, 26.101082, 0.0],
                        [-1559.157471, -30.753305, 0.0],
                        [-1520.886353,  -30.761044, 0.0],
                        [-1561.039063, 31.522200, 0.0],
                        [-1561.039795, -33.577713, 0.0]]
                        # [-600.0, 31.5, 0.0],
                        # [-600.0, -23.5, 0.0]]
        self.state = None
        self.estimated_state = None

    def state_callback(self, msg):
        pos = msg.pose[1].position
        ori = msg.pose[1].orientation
        self.state = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    def image_callback(self, img_msg):
        # Try to convert the ROS image to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img = PILImage.fromarray(cv_image, mode='RGB')

            output = self.predict_img(img)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        

        # Key points detection using trained nn.
        keypoints = []
        for i in range(14):
            p = (((output[0,i,:,:]==torch.max(output[0,i,:,:])).nonzero())/args.scale).tolist()
            p[0].reverse()
            keypoints.append(p[0])

        # # Pose estimation using PnP
        # self.show_image(cv_image, keypoints)
        success, rotation_vector, translation_vector = self.pose_estimation(np.array(keypoints), np.array(self.keypoints), self.K)
        if success:
            Rot = cv2.Rodrigues(rotation_vector)[0]
            RotT = np.matrix(Rot).T
            camera_position = -RotT*np.matrix(translation_vector)

            R = Rot
            sin_x = sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
            singular  = sin_x < 1e-6
            if not singular:
                z1 = atan2(R[2,0], R[2,1])     # around z1-axis
                x = atan2(sin_x,  R[2,2])     # around x-axis
                z2 = atan2(R[0,2], -R[1,2])    # around z2-axis
            else:
                z1 = 0                                         # around z1-axis
                x = atan2(sin_x,  R[2,2])     # around x-axis
                z2 = 0                                         # around z2-axis

            angles = np.array([[z1], [x], [z2]])
            yawpitchroll_angles = -angles
            yawpitchroll_angles[0,0] = (yawpitchroll_angles[0,0] + (5/2)*pi)%(2*pi) # change rotation sense if needed, comment this line otherwise
            yawpitchroll_angles[1,0] = -(yawpitchroll_angles[1,0]+pi/2)
            if yawpitchroll_angles[0,0] > pi:
                yawpitchroll_angles[0,0] -= 2*pi

            self.estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[2,0], yawpitchroll_angles[1,0], yawpitchroll_angles[0,0]]
        else:
            print("Pose Estimation Failed.")

        self.show_image(cv_image, keypoints)
    def show_image(self, cv_image, keypoints):
        kp_img = cv_image
        for i in range(len(keypoints)):
           kp_img = cv2.circle(kp_img, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(3)

    def predict_img(self, full_img, scale_factor=0.5, out_threshold=0.5):
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()

            return output 


    def mask_to_image(self, mask: np.ndarray, mask_values):
        if isinstance(mask_values[0], list):
            out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
        elif mask_values == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
        else:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

        if mask.ndim == 3:
            mask = np.argmax(mask, axis=0)

        for i, v in enumerate(mask_values):
            out[mask == i] = v

        return Image.fromarray(out)
    
    def detect_keypoints(self, net, img, device):
        mask = self.predict_img(net=net, full_img=img, device=device)

        return 

    def pose_estimation(self, image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        return cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

def run_controller(x_cur, x_ref, v_max=10, acc_max=1, beta_max=1, omega_max=1):
    model = aircraft_model(x_ref)
    mpc = aircraft_mpc(model, v_max, acc_max, beta_max, omega_max)
    simulator = aircraft_simulator(model)
    simulator.x0 = x_cur
    mpc.x0 = x_cur

    u_init = np.full((3, 1), 0.0)
    mpc.u0 = u_init
    simulator.u0 = u_init
    mpc.set_initial_guess()

    u0 = mpc.make_step(x_cur)
    x_next = simulator.make_step(u0)

    return x_next

def path(t, x = -3000.0, y = 0, z = 100, yaw = 0):
    approaching_angle = 3
    k = np.tan(approaching_angle*(pi/180))
    v = 100.0
    # Straight line along x-axis
    if  z - k*v*t <= 0:
        return  [cos(yaw)*(v*t) + x, (sin(yaw)*(v*t) + y), 0, yaw], [-v, 0.0, 0]
    return [cos(yaw)*(v*t) + x, (sin(yaw)*(v*t) + y), z - k*v*t, yaw], [-v, -k*v, 0]

def create_state_msd(x, y, z, roll, pitch, yaw):
    state_msg = ModelState()
    state_msg.model_name = 'fixedwing'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z

    q = squaternion.Quaternion.from_euler(roll, pitch, yaw)
    
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w

    return state_msg

data_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel'

def update_aircraft_position(net, device):
    initial_state = [-3000.0, 0, 100, 0, 0, 0]

    perception = Perception(net, device)
    init_msg = create_state_msd(initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4], initial_state[5])
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(init_msg)
    except rospy.ServiceException:
        print("Service call failed")

    cur_state = [-3000.0, 0, 100, 0, 0, 0]
    
    idx = 0

    true_states = []
    estimated_states = []
    cur_time = 0
    while not rospy.is_shutdown():
        ref_state, _ = path(cur_time)
        cur_time += 0.1
        cur_state = run_controller(np.array(cur_state), [ref_state[0], ref_state[1], ref_state[2], 0, -3*(pi/180), ref_state[3]])
        state_msg = create_state_msd(cur_state[0], cur_state[1], cur_state[2], 0, cur_state[4], cur_state[5])

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException:
            print("Service call failed")

        if perception.estimated_state is None:
            continue

        time.sleep(0.01)

        
        true_state = [cur_state[0], cur_state[1], cur_state[2], cur_state[3], cur_state[4], cur_state[5]]
        # print("State: ", true_state)
        # print("Estimation: ", perception.estimated_state)
        true_states.append(true_state)
        estimated_states.append(perception.estimated_state)
        perception.count = idx
        # idx += 1
        np.save("ground_truth", np.array(true_states))
        np.save("estimation", np.array(estimated_states))

import argparse
import logging
def get_args():
    model = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/model.pth'
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
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('update_poses', anonymous=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # net = UNet(in_channels=3, out_classes=16)
    net = UNet(n_channels=3, n_classes=16, bilinear=args.bilinear)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    try:
        update_aircraft_position(net, device)
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
        
