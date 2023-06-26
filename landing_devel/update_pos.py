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

import aircraft_controller

# Path to the image directory. 
img_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/imgs'

body_height = 0.77
pitch_offset = 0
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

            # cv_image = self.add_noise_to_image(cv_image)
            
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

            self.estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item() - body_height, yawpitchroll_angles[2], yawpitchroll_angles[1] - pitch_offset, yawpitchroll_angles[0]]
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



def integrator_dynamics(x_ground_truth, t, x_ref, x_estimated):
    x_ground_truth = np.array(x_ground_truth)
    u = compute_control(x_estimated, x_ref)
    # dot_x = np.array(((x_ground_truth[6:]).tolist()).extend(list(u)))
    dot_x = np.array([x_ground_truth[6], x_ground_truth[7], x_ground_truth[8], x_ground_truth[9], x_ground_truth[10], x_ground_truth[11], u[0], u[1], u[2], u[3], u[4], u[5]])
    return dot_x

A = np.zeros((12, 12))
B = np.zeros((12, 6))
for i in range(6):
    A[i, 6+i] = 1
    B[6+i, i] = 1
# print(A, B)
Q = np.eye(12)
R = np.eye(6)
K, S, E = control.lqr(A, B, Q, R)


def compute_control(x_cur, x_ref):
    u = -K.dot(x_cur - x_ref)
    return u

def run_PD_controller(x_ref, x_cur_estimated, x_cur_ground_truth, dt):
    # A = np.zeros((6, 6))
    # B = np.zeros((6, 3))
    # sin_psi = sin(x_ref[4])
    # cos_psi = cos(x_ref[4])
    # cos_theta = cos(x_ref[5])
    # sin_theta = sin(x_ref[5])
    # v_0 = x_ref[3]

    # A[0, 3] = cos_psi*cos_theta
    # A[0, 4] = -v_0*sin_psi*cos_theta
    # A[0, 5] = -v_0*cos_psi*sin_theta

    # A[1, 3] = sin_psi*cos_theta
    # A[1, 4] = v_0*cos_psi*cos_theta
    # A[1, 5] = -v_0*sin_psi*sin_theta
    
    # A[2, 3] = sin_theta
    # A[2, 5] = v_0*cos_theta

    # B[3, 0] = 1
    # B[4, 1] = 1
    # B[5, 2] = 1 

    # tf = scipy.signal.ss2tf(A, B, np.eye(6), np.zeros((6, 3)))
    # print("Transfer function: ", tf)
    # K, S, E = control.lqr(A, B, 10.0*np.eye(6), np.eye(3))
    # return np.matmul(K, x_cur - x_ref)

    res = odeint(integrator_dynamics, x_cur_ground_truth, [0, dt], args=(x_ref, x_cur_estimated))[-1]
    return res

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

# def path(initial_state, cur_state, ref_speed, approaching_angle, yaw = 0, dt=0.5):
#     '''
#     Planner.
#     Generate path for the aircraft to follow.
#     '''
#     k = np.tan(approaching_angle*(pi/180))
#     # Reference speed.
#     v = ref_speed

#     return [v*dt + cur_state[0], 0, cur_state[2] - k*v*dt, yaw]

def generate_path(initial_state, approaching_angle):
    x_total = initial_state[0]
    z_total = initial_state[2]

    k = np.tan(approaching_angle*(pi/180))
    delta_x = 50
    delta_z = k*delta_x
    waypoints = []
    
    while z_total - delta_z > 0 and x_total + delta_x < -1566:
        x_total += delta_x
        z_total -= delta_z
        waypoints.append([x_total, 0, z_total])

    # print("WAYPOINTS: ", waypoints)
    return np.array(waypoints)

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
    initial_state = [-2500.0, 0, 120.0, 0, -np.deg2rad(3), 0]

    # One simulation length.
    delta_t = 0.5
    # Reference speed
    ref_speed = 50.0
    # Angle of the aircraft relative to the ground (in degrees).
    approaching_angle = 3

    # Perception module.
    perception = Perception(net, device)
    init_msg = create_state_msd(initial_state[0], initial_state[1], initial_state[2], initial_state[3], -initial_state[4], initial_state[5])
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

    estimated_rate = np.zeros(6)
    true_rate = np.zeros(6)
    # Surrogate model
    # surrogate_model = tf.keras.models.load_model('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/surrogate_model/model.keras')

    # Ground truth.
    true_states = []
    true_states.append(np.array(initial_state))
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
    # y_average = np.zeros(50)


    T = 1
    # set_new_state_flag = False
    set_new_state_counter = 0
    
    # Computed waypoints
    waypoints = generate_path(initial_state, approaching_angle)
    waypoints_index = 0

    # Count inaccurate estimation.
    corrupted_count = 0

    while not rospy.is_shutdown():
        time.sleep(0.1)
        if perception.estimated_state is None or len(perception.estimated_state) == 0:
            continue
        estimated_state = perception.estimated_state
        estimated_state[4] = -estimated_state[4]

        if len(estimated_states) > 0 and len(true_states) > 0:
            for i in range(6):
                estimated_rate[i] = (estimated_states[-1][i] - estimated_state[i])/delta_t
                true_rate[i] = (true_states[-1][i] - true_state[i])/delta_t


        '''
        Planner
        '''
        # ref_state = path(initial_state, estimated_state, ref_speed, approaching_angle, dt=delta_t)
        if np.linalg.norm(np.array(estimated_state[:3]) - np.array(waypoints[waypoints_index])) < 10:
            waypoints_index += 1
        if waypoints_index >= len(waypoints):
            break

        ref_state = waypoints[waypoints_index]
        ref_states.append(ref_state)

        cur_time += delta_t


        '''
        MPC
        '''
        x_ground_truth = np.array([true_state[0], true_state[1], true_state[2], np.linalg.norm(true_rate[:3]), true_state[5], true_state[4]])
        x_estimated = np.array([estimated_state[0], estimated_state[1], estimated_state[2], np.linalg.norm(estimated_rate[:3]), estimated_state[5], estimated_state[4]])
        if np.linalg.norm(x_ground_truth - x_estimated) > 50:
            print(">>>>>> Estimated Corrupted ", x_estimated)
            x_estimated = x_ground_truth 
            corrupted_count += 1
        x_next = run_controller(x_ground_truth, x_estimated, ref_state, delta_t)
        cur_state = np.array([x_next[0], x_next[1], x_next[2], 0, x_next[5], x_next[4]])
        '''
        Quadrotor control.
        '''
        # x_ground_truth = np.array([true_state[0], true_rate[0], true_state[1], true_rate[1], true_state[2], true_rate[2], true_state[3], true_rate[3], true_state[4], true_rate[4], true_state[5], true_rate[5]])
        # x_estimated = np.array([estimated_state[0], estimated_rate[0], estimated_state[1], estimated_rate[1], estimated_state[2], estimated_rate[2], estimated_state[3], estimated_rate[3], estimated_state[4], estimated_rate[4], estimated_state[5], estimated_rate[5]])
        # # x_estimated = np.array([estimated_state[0], estimated_rate[0], estimated_state[1], estimated_rate[1], estimated_state[2], estimated_rate[2], true_state[3], true_rate[3], true_state[4], true_rate[4], true_state[5], true_rate[5]])
        # x_estimated = x_ground_truth
        # # print(">>>> Estimated", x_estimated)
        # # if np.linalg.norm(x_ground_truth - x_estimated) > 50:
        #     # print(">>>>>> Estimated Corrupted ", x_estimated)
        #     # x_estimated = x_ground_truth 
        # # print(">>>> Ground Truth", x_ground_truth)
        # # print(">>>> Estimated", x_estimated)
        # next_state = aircraft_controller.simulate(x_ground_truth, x_estimated, np.array([ref_state[0], ref_state[1], ref_state[2]]), delta_t)
        # # print(">>>> Next state", next_state)
        # next_state[7] = (next_state[6] - x_ground_truth[6])/delta_t
        # next_state[9] = (next_state[8] - x_ground_truth[8])/delta_t
        # next_state[11] = (next_state[10] - x_ground_truth[10])/delta_t

        # cur_state = np.array([next_state[0], next_state[2], next_state[4], next_state[6], next_state[8], next_state[10]])
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        '''
        Integrator control.
        '''
        # print("Ground Truth: ", true_state)
        # print("Estimated State: ", perception.estimated_state)
        # print((ref_state.tolist()).extend([0, 0, 0, 0, 0, 0]))
        # x_ref = (ref_state.tolist()).extend([0, 0, 0, 0, 0, 0])
        # x_ref = np.array([ref_state[0], ref_state[1], ref_state[2], 0, -np.deg2rad(3), 0, 0, 0, 0, 0, 0, 0])
        # x_ground_truth = np.concatenate((true_state, true_rate))
        # x_estimated = np.concatenate((estimated_state, estimated_rate))
        # if np.linalg.norm(x_ground_truth - x_estimated) > 50:
        #     print(">>>>>> Estimation Corrupted ", x_estimated)
        #     x_estimated = x_ground_truth    
        # # x_estimated = np.array([estimated_state[0], estimated_rate[0], estimated_state[1], estimated_rate[1], estimated_state[2], estimated_rate[2], true_state[3], true_rate[3], true_state[4], true_rate[4], true_state[5], true_rate[5]])
        # # x_estimated = x_ground_truth

        # next_state = run_PD_controller(x_ref, x_estimated, x_ground_truth, delta_t)
        # cur_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5]])
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        
        state_msg = create_state_msd(cur_state[0], cur_state[1], cur_state[2], cur_state[3], -cur_state[4], cur_state[5])
    
        nominal_states.append([ref_speed*delta_t + cur_state[0], 0, cur_state[2] - np.tan(np.deg2rad(approaching_angle))*ref_speed*delta_t, 0])
        
        if set_new_state_counter % T == 0:
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state( state_msg )

            except rospy.ServiceException:
                print("Service call failed")


            time.sleep(0.5)


        true_state = cur_state
        # print("Ground Truth: ", true_state)
        # print("Estimated State: ", perception.estimated_state)
        true_states.append(true_state)
        estimated_states.append(estimated_state)
        # averaged_states.append(np.mean(y_average))
        # estimated_states_surrogate.append(surrogate_model.predict(true_state))
        # perception.count = idx
        idx += 1
        set_new_state_counter += 1

        
        np.save("ground_truth", np.array(true_states))
        np.save("estimation", np.array(estimated_states))
        # np.save("ref_states", np.array(ref_states))
        # np.save("averaged_states", np.array(averaged_states))
        # np.save("nominal_states", np.array(nominal_states))
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
        
