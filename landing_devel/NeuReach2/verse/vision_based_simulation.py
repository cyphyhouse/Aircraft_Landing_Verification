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

import pickle
import matplotlib.pyplot as plt 
from std_msgs.msg import ColorRGBA

# from scipy.spatial.transform import Rotation 
# import matplotlib.pyplot as plt 

# from aircraft_model import *
# from aircraft_mpc import *
# from aircraft_simulator import *

# import control
# from scipy.integrate import odeint

# import aircraft_controller
from fixed_wing_agent3 import FixedWingAgent3
from verse import Scenario, ScenarioConfig

import copy

import argparse
import logging
from enum import Enum, auto
from gazebo_msgs.srv import SetModelState, SetLightProperties

def set_light_properties(light_value: float) -> None:
    GZ_SET_LIGHT_PROPERTIES = "/gazebo/set_light_properties"
    rospy.wait_for_service(GZ_SET_LIGHT_PROPERTIES)
    try:
        set_light_properties_srv = \
            rospy.ServiceProxy(GZ_SET_LIGHT_PROPERTIES, SetLightProperties)
        resp = set_light_properties_srv(
            light_name='sun',
            cast_shadows=True,
            diffuse=ColorRGBA(int(204*light_value),int(204*light_value),int(204*light_value),255),
            specular=ColorRGBA(51, 51, 51, 255),
            attenuation_constant=0.9,
            attenuation_linear=0.01,
            attenuation_quadratic=0.0,
            direction=Vector3(-0.483368, 0.096674, -0.870063),
            pose=Pose(position=Point(0, 0, 10), orientation=Quaternion(0, 0, 0, 1))
        )
        # TODO Check response
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s" % e)


class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

script_dir = os.path.dirname(os.path.realpath(__file__)) 
img_path = os.path.join(script_dir, '../../imgs')

def get_args():
    model = os.path.join(script_dir, '../../model.pth')
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

def sample_point(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.random.uniform(low, high) 

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

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

body_height = 0.77
pitch_offset = 0

class Perception:
    def __init__(self, net, device):
        self.net = net 
        self.device = device
        self.image_updated = False
        self.image = None
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()

        # Camera matrix (intrinsic matrix)
        self.K = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])

        # The set of key points used for state estimation.
        self.keypoints = [[-1221.370483, 16.052534, 5.0],
                        [-1279.224854, 16.947235, 5.0],
                        [-1279.349731, 8.911615, 5.0],
                        [-1221.505737, 8.033512, 5.0],
                        [-1221.438110, -8.496282, 5.0],
                        [-1279.302002, -8.493725, 5.0],
                        [-1279.315796, -16.504263, 5.0],
                        [-1221.462402, -16.498976, 5.0],
                        [-1520.81, 26.125700, 5.0],
                        [-1559.122925, 26.101082, 5.0],
                        [-1559.157471, -30.753305, 5.0],
                        [-1520.886353,  -30.761044, 5.0],
                        [-1561.039063, 31.522200, 5.0],
                        [-1561.039795, -33.577713, 5.0]]
    
        self.state = None
        self.error_idx = []
        self.idx = None

    def state_callback(self, msg):
        pos = msg.pose[1].position
        ori = msg.pose[1].orientation
        self.state = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
 
    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.image = copy.deepcopy(cv_image)
            self.image_updated = True
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

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

    def show_image(self, cv_image, keypoints):
        kp_img = cv_image
        for i in range(len(keypoints)):
           kp_img = cv2.circle(kp_img, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))

        # kp_img = self.add_noise_to_image(kp_img)
  
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(3)

    def vision_estimation(self, cv_image):
        # Try to convert the ROS image to a CV2 image
        img = PILImage.fromarray(cv_image, mode='RGB')
        
        # Get probabilistic heat maps corresponding to the key points.
        output = self.predict_img(img)

        # plt.figure(0)
        # plt.imshow(img)
        # plt.figure(1)
        # plt.imshow(np.sum(output[0].detach().numpy(),axis=0))
        # plt.show()

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

    def wait_img_update(self):
        while not self.image_updated:
            time.sleep(0.1)

    def set_percept(self, point: np.ndarray) -> np.ndarray:
        # Set aircraft to pos
        self.set_pos(point)

        self.wait_img_update()

        img = self.image
        self.vision_estimation(img)

        estimated_state = np.array([
            self.estimated_state[0],
            self.estimated_state[1],
            self.estimated_state[2],
            self.estimated_state[5],
            self.estimated_state[4],
            point[5]
        ])

        # if np.linalg.norm(estimated_state - point) > 50:
        #     print(">>>>>> Estimated Corrupted ", estimated_state)
        #     estimated_state = point 
        #     self.error_idx.append(self.idx)
        

        return estimated_state

    def set_pos(self, point: np.ndarray) -> np.ndarray:
        # Set aircraft to given pose
        init_msg = create_state_msd(point[0], point[1], point[2], 0, point[4], point[3])
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # Set initial state.
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(init_msg)
        except rospy.ServiceException:
            print("Service call failed")
 

def run_vision_sim(scenario, init_point, init_ref, time_horizon, computation_step, time_step, vision_estimator: Perception):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    estimate_traj = []
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_point = vision_estimator.set_percept(point)
        estimate_traj.append(estimate_point)
        # estimate_point = sample_point(estimate_lower, estimate_upper)
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
    return traj, estimate_traj


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
    fixed_wing_controller = os.path.join(script_dir, './fixed_wing3_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    
    vision = Perception(net, device)

    

    state = np.array([
        [-3050.0, -20, 110.0, 0-0.01, -np.deg2rad(3)-0.01, 10-0.1], 
        [-3010.0, 20, 130.0, 0+0.01, -np.deg2rad(3)+0.01, 10+0.1]
    ])
    
    traj_list = []
    estimate_traj_list = []
    init_list = []
    for i in range(200):
        vision.idx=i
        init_point = sample_point(state[0,:], state[1,:])
        init_ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
        time_horizon = 100

        e = np.random.uniform(0.5, 1.2)
        set_light_properties(e)
        init_list.append((init_point, [e]))

        try:
            # Run simulation.
            traj, estimate_traj = run_vision_sim(fixed_wing_scenario, init_point, init_ref, time_horizon, 0.1, 0.01, vision)
            traj_list.append(traj) 
            estimate_traj_list.append(estimate_traj) 
        except rospy.exceptions.ROSInterruptException:
            rospy.loginfo("Stop updating aircraft positions.")
            
        with open('vcs_sim.pickle','wb+') as f:
            pickle.dump(traj_list, f)
        with open('vcs_estimate.pickle','wb+') as f:
            pickle.dump(estimate_traj_list, f)
        with open('vcs_init.pickle', 'wb+') as f:
            pickle.dump(init_list, f)

    print(vision.error_idx)