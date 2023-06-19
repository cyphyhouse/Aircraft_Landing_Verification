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

# Path to the image directory. 
img_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/imgs'

class Perception():
    '''
    Perception module
    '''
    def __init__(self, net, device):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()

        # Camera matrix (intrinsic matrix)
        self.K = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])

        self.net = net
        self.device = device

        # The set of key points used for state estimation.
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
        # Latest ground truth of the aircraft state.
        self.state = None
        # Latest estimation of the aircraft state.
        self.estimated_state = None
        # Counter of the number of simulations.
        self.count = 0 

    def state_callback(self, msg):
        pos = msg.pose[1].position
        ori = msg.pose[1].orientation
        self.state = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    def image_callback(self, img_msg):
        # Try to convert the ROS image to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            cv_image = self.add_noise_to_image(cv_image)
            
            img = PILImage.fromarray(cv_image, mode='RGB')
            
            # Get probabilistic heat maps corresponding to the key points.
            output = self.predict_img(img)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        

        # Key points detection using trained nn. Extract pixels with highest probability.
        keypoints = []
        for i in range(14):
            p = (((output[0,i,:,:]==torch.max(output[0,i,:,:])).nonzero())/args.scale).tolist()
            p[0].reverse()
            keypoints.append(p[0])

        # Pose estimation via PnP.
        success, rotation_vector, translation_vector = self.pose_estimation(np.array(keypoints), np.array(self.keypoints), self.K)
        if success:
            # TODO: THIS PART NEEDS TO BE REVISED.
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

            angles = np.array([z1, x, z2])
            yawpitchroll_angles = -angles
            yawpitchroll_angles[0] = (yawpitchroll_angles[0] + (5/2)*pi)%(2*pi) # change rotation sense if needed, comment this line otherwise
            yawpitchroll_angles[1] = -(yawpitchroll_angles[1]+pi/2)
            if yawpitchroll_angles[0] > pi:
                yawpitchroll_angles[0] -= 2*pi

            self.estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[2], yawpitchroll_angles[1], yawpitchroll_angles[0]]
        else:
            print("Pose Estimation Failed.")

        self.show_image(cv_image, keypoints)

    def add_noise_to_image(self, image):
        '''
        Source: https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
        '''
        brightness = 30
        contrast = 30
        image = np.int16(image)
        image = image * (contrast/127+1) - contrast + brightness
        image = np.clip(image, 0, 255)
        image = np.uint8(image)

        # ksize
        ksize = (10, 10)
        
        # Using cv2.blur() method 
        image = cv2.blur(image, ksize) 
        return image

    def show_image(self, cv_image, keypoints):
        kp_img = cv_image
        for i in range(len(keypoints)):
           kp_img = cv2.circle(kp_img, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))

        # kp_img = self.add_noise_to_image(kp_img)
  
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(3)

    def predict_img(self, full_img, scale_factor=1.0, out_threshold=0.5):
        '''
        Unet.
        '''
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()

            return output 

    def pose_estimation(self, image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        '''
        Pose estimation via solvePnP.
        '''
        return cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

def run_controller(x_true, x_cur, x_ref, delta_t, v_max=50, acc_max=20, beta_max=0.02, omega_max=0.03):
    '''
    Controller.
    x_true: ground truth of current state.
    x_cur: estimation of current state.
    x_ref: reference waypoint.
    v_max: maximum speed.
    acc_max: maximum acceleration.
    beta_max: maximum yaw rate.
    omage_max: maximum pitch rate.
    '''
    model = aircraft_model(x_ref)
    mpc = aircraft_mpc(model, v_max, acc_max, beta_max, omega_max, delta_t)
    simulator = aircraft_simulator(model, delta_t)
    simulator.x0 = np.array(x_true)
    mpc.x0 = x_cur

    u_init = np.full((3, 1), 0.0)
    mpc.u0 = u_init
    simulator.u0 = u_init
    mpc.set_initial_guess()

    u0 = mpc.make_step(x_cur)
    x_next = simulator.make_step(u0)

    return x_next

def path(cur_state, ref_speed, approaching_angle, yaw = 0, dt=0.05):
    '''
    Planner.
    Generate path for the aircraft to follow.
    '''
    k = np.tan(approaching_angle*(pi/180))
    # Reference speed.
    v = ref_speed

    return [v*dt + cur_state[0], 0, cur_state[2] - k*v*dt, yaw]


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
    '''
    Main function.
    '''
    # Initial state of the aircraft.
    initial_state = [-2500.0, 00, 78.6, 0, 0, 0]

    # One simulation length.
    delta_t = 0.02
    # Reference speed
    ref_speed = 50.0
    # Angle of the aircraft relative to the ground (in degrees).
    approaching_angle = 3

    # Perception module.
    perception = Perception(net, device)
    init_msg = create_state_msd(initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4], initial_state[5])
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        # Set initial state.
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(init_msg)
    except rospy.ServiceException:
        print("Service call failed")

    cur_state = initial_state
    true_state = initial_state
    idx = 0

    # Surrogate model
    # surrogate_model = tf.keras.models.load_model('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/surrogate_model/model.keras')

    # Ground truth.
    true_states = []
    # Estimated states.
    estimated_states = []
    # Filtered states.
    averaged_states = []
    # Reference states. (waypoints)
    ref_states = []
    # Reference states projected to the nomial landing trajectory.
    nominal_states = []
    # # Estimated states given by the surrogate model.
    # estimated_states_surrogate = []
    # Time.
    cur_time = 0
    # Average y coordinates.
    y_average = np.zeros(50)

    while not rospy.is_shutdown():
        time.sleep(0.01)
        if perception.estimated_state is None or len(perception.estimated_state) == 0:
            continue
        estimated_state = perception.estimated_state

        ref_state = path(estimated_state, ref_speed, approaching_angle)
        ref_states.append(ref_state)

        cur_time += delta_t

        cur_state = run_controller(true_state, np.array(estimated_state), [ref_state[0], ref_state[1], ref_state[2], -3*(pi/180)], delta_t)
        state_msg = create_state_msd(cur_state[0], cur_state[1], cur_state[2], 0, cur_state[5], cur_state[4])
    
        nominal_states.append([ref_speed*delta_t + cur_state[0], 0, cur_state[2] - np.tan(np.deg2rad(approaching_angle))*ref_speed*delta_t, 0])
        
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException:
            print("Service call failed")


        time.sleep(0.01)

        
        true_state = [cur_state[0], cur_state[1], cur_state[2], 0, cur_state[5], cur_state[4]]
        print("Ground Truth: ", true_state)
        print("Estimated State: ", perception.estimated_state)
        true_states.append(true_state)
        estimated_states.append(perception.estimated_state)
        averaged_states.append(np.mean(y_average))
        # estimated_states_surrogate.append(surrogate_model.predict(true_state))
        perception.count = idx
        idx += 1
        
        np.save("ground_truth", np.array(true_states))
        np.save("estimation", np.array(estimated_states))
        np.save("ref_states", np.array(ref_states))
        np.save("averaged_states", np.array(averaged_states))
        np.save("nominal_states", np.array(nominal_states))
        # np.save("surrogate_model_predicted_states", np.array(estimated_states_surrogate))
        # if true_state[0] > -2350:
        #     break

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
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

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

    try:
        # Run simulation.
        update_aircraft_position(net, device)
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
        
