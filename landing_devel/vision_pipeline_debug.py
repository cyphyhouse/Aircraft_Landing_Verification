import torch
import torch.nn.functional as F
from unet import UNet
from utils.data_loading import BasicDataset
import os 

import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
import math
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
import PIL 
from scipy.spatial.transform import Rotation 
import os 
import matplotlib.pyplot as plt 

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, SetLightProperties
from rosplane_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA

# from aircraft_model import *
# from aircraft_mpc import *
# from aircraft_simulator import *

from PIL import Image as PILImage

script_dir = os.path.realpath(os.path.dirname(__file__))
data_path = os.path.join(script_dir, 'data')
label_path = os.path.join(script_dir, 'estimation_label')

keypoints = [[-1221.370483, 16.052534, 5.0],
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

state = [-3001.4117941120794, -14.106879907553605, 61.10726693286948, 0.018147651076384247, -0.03218774235308877, 0.09911239499369592]

GZ_SET_LIGHT_PROPERTIES = "/gazebo/set_light_properties"

cv_bridge = CvBridge()
cv_img = None
def image_callback(img_msg):
    global cv_img
    cv_img = cv_bridge.imgmsg_to_cv2(img_msg, "passthrough")
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

K = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])
def create_state_msd(x, y, z, roll, pitch, yaw):
    state_msg = ModelState()
    state_msg.model_name = 'fixedwing'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z

    # print(roll, pitch, yaw)
    q = squaternion.Quaternion.from_euler(roll, pitch, yaw)
    
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w

    return state_msg

def convert_to_image(world_pos, ego_pos, ego_ori):
    objectPoints = np.array(world_pos) 
    R = Rotation.from_quat(ego_ori)
    R2 = Rotation.from_euler('xyz',[-np.pi/2, -np.pi/2, 0])
    R_roted = R2*R.inv()

    #TODO: The way of converting rvec is wrong
    rvec = R_roted.as_rotvec()
    tvec = -R_roted.apply(np.array(ego_pos))
    cameraMatrix = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pnt,_ = cv2.projectPoints(objectPoints,rvec,tvec,cameraMatrix,distCoeffs)
    return pnt  

def pose_estimation(image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
    '''
    Pose estimation via solvePnP.
    '''
    return cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

def state_estimation_func(rotation_vector, translation_vector):
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

    return [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[2,0], yawpitchroll_angles[1,0], yawpitchroll_angles[0,0]]

def check_valid_img(success, rotation_vector, translation_vector, state_ground_truth, tolr=1):
    if success:
        estimated_state = state_estimation_func(rotation_vector,translation_vector)
        estimation_error = np.linalg.norm(np.array(estimated_state) - np.array(state_ground_truth))
        print("State estimation: ", estimated_state)
        print("State ground truth: ", state_ground_truth)
        if 0.5*(abs(estimated_state[0] - state_ground_truth[0]) + abs(estimated_state[1] - state_ground_truth[1]) + abs(estimated_state[2] - state_ground_truth[2])) + (abs(estimated_state[3] - state_ground_truth[3]) + abs(estimated_state[4] - state_ground_truth[4]) + abs(estimated_state[5] - state_ground_truth[5])) > 1:
            return False, estimation_error, estimated_state
        return True, estimation_error, estimated_state
    else:
        return False, math.inf, []

def predict_img(full_img, scale_factor=1.0, out_threshold=0.5):
    '''
    Unet.
    '''
    global net, device
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()

        return output 

def state_estimator(cv_image):
    try:
        img = PILImage.fromarray(cv_image, mode='RGB')
        # Get probabilistic heat maps corresponding to the key points.
        output = predict_img(img)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        
    # Key points detection using trained nn. Extract pixels with highest probability.
    keypoints_detected = []
    for i in range(14):
        p = (((output[0,i,:,:]==torch.max(output[0,i,:,:])).nonzero())/args.scale).tolist()
        p[0].reverse()
        p = p[0]
        # p[0] = 640 - p[0]
        # p[1] = 480 - p[1]
        keypoints_detected.append(p)
    
    output_vis = output.numpy().squeeze()

    kp_img = cv_img 
    for kp in keypoints_detected:
        kp_img = cv2.circle(kp_img, (np.int32(kp)[0],np.int32(kp)[1]), 5, (0,0,255), 2)
    
    # alpha = 0.5
    # beta = 10.0

    # adjusted = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)

    plt.figure()
    plt.imshow(cv_img)

    plt.figure()
    for i in range(14):
        output_vis_sum = np.sum(output_vis, axis=0)
        plt.imshow(output_vis_sum[:,:])
    plt.show()
    
    success, rotation_vector, translation_vector = pose_estimation(np.array(keypoints_detected), np.array(keypoints), K)
    if success:
        # TODO: THIS PART NEEDS TO BE REVISED.
        estimated_state = state_estimation_func(rotation_vector, translation_vector)
        return estimated_state, keypoints_detected, rotation_vector, translation_vector
    else:
        print("Pose Estimation Failed.")
        return None, None, None, None


def set_light_properties(light_value: float) -> None:
    
    light_condition = 204*light_value 
    if light_condition > 255:
        light_condition = 255
    rospy.wait_for_service(GZ_SET_LIGHT_PROPERTIES)
    try:

        set_light_properties_srv = \
            rospy.ServiceProxy(GZ_SET_LIGHT_PROPERTIES, SetLightProperties)
        resp = set_light_properties_srv(
            light_name='sun',
            cast_shadows=True,
            diffuse=ColorRGBA(int(light_condition),int(light_condition),int(light_condition),255),
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

def sample_state_estimator():
    global cv_img
    rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, image_callback)
    # Predicted path that the agent will be following over the time horizon

    light_value = 0.1
    print(f"Light level: {light_value}")
    set_light_properties(light_value)
    time.sleep(0.1)

    idx = 0
    state_rand = [-3000.0, 0.0, 120.0, 0.0, 0.0, 0.0]
    state_msg = create_state_msd(state_rand[0], state_rand[1], state_rand[2], state_rand[3], state_rand[4], state_rand[5])

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        _ = set_state( state_msg )
    except rospy.ServiceException:
        print("Service call failed")

    time.sleep(0.1)
    q = squaternion.Quaternion.from_euler(state_rand[3], state_rand[4], state_rand[5])
    offset_vec = np.array([-1.1*0.20,0,0.8*0.77])
    aircraft_pos = np.array([state_rand[0], state_rand[1], state_rand[2]]) # x,y,z
    aircraft_ori = [q.x, q.y, q.z, q.w] # x,y,z,w
    R = Rotation.from_quat(aircraft_ori)
    aircraft_pos += offset_vec

    keypoint_position_in_image = []
    for i in range(len(keypoints)):
        image_coordinate = convert_to_image(keypoints[i], aircraft_pos, aircraft_ori).flatten()
        image_coordinate[0] = 640-image_coordinate[0]
        image_coordinate[1] = 480-image_coordinate[1]
        if image_coordinate[0] > 640 or image_coordinate[0] < 0 or image_coordinate[1] > 480 or image_coordinate[1] < 0:
            break
        keypoint_position_in_image.append(image_coordinate.tolist())
    if len(np.array(keypoint_position_in_image)) < 14:
        return
    # print(keypoint_position_in_image)
    # print(np.array(keypoints).shape)
    success, rotation_vector, translation_vector = pose_estimation(np.array(keypoint_position_in_image), np.array(keypoints), K)
    valid_img, error, estimated_state = check_valid_img(success, rotation_vector, translation_vector, state_rand)
    # q_estimated = squaternion.Quaternion.from_euler(estimated_state[3], estimated_state[4], estimated_state[5])
    if not valid_img or cv_img is None:
        return

    # cv2.imshow('camera', cv_img)
    # cv2.waitKey(3)
    # for i in range(10):
    state_estimation, kps, est_rot, est_trans = state_estimator(cv_img)
    # kp_img = cv_img 
    # for kp in kps:
    #     kp_img = cv2.circle(kp_img, (np.int32(kp)[0],np.int32(kp)[1]), 5, (0,0,255), 2)
    # cv2.imshow('camera', cv_img)
    # cv2.waitKey(3)
    
    if 0.5*(abs(state_estimation[0] - state_rand[0]) + abs(state_estimation[1] - state_rand[1]) + abs(state_estimation[2] - state_rand[2])) + (abs(state_estimation[3] - state_rand[3]) + abs(state_estimation[4] - state_rand[4]) + abs(state_estimation[5] - state_rand[5])) > 100:
        print("stop here")

    print("Estimated state: ", state_estimation)
    
import argparse
import logging
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

if __name__ == "__main__":
    global net, device
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
        sample_state_estimator()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
        
        
