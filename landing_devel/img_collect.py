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
import PIL 

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from rosplane_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from sensor_msgs.msg import Image

# import torch
# import torch.nn.functional as F
# from unet import UNet
# from utils.data_loading import BasicDataset
import os
img_path = '/home/younger/work/Aircraft_landing_verification/src/landing_devel/imgs'
data_path = '/home/younger/work/Aircraft_landing_verification/src/landing_devel/data'
# img_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/imgs'
class Perception():
    def __init__(self, name=''):
        self.name = name

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()
        # cv2.namedWindow("Image Window", 1)

        self.count = -1
        self.pic_index = 0
        # Camera matrix
        self.K = np.array([[205.46963709898583, 0.0, 320.5],
                    [0.0, 205.46963709898583, 240.5],
                    [0.0, 0.0, 1.0]])

        self.keypoints = {'mask1': np.array([[1512.482910, 19.944490, 0.0], 
                                            [1533.836304, 19.864525, 0.0],
                                            [1533.663940, 59.833881, 0.0],
                                            [1512.343628, 60.201397, 0.0]]),
                        'mask2': np.array([[1288.145264, 27.234539, 0.0], 
                                            [1311.588867, 27.199703, 0.0],
                                            [1311.613647, 33.277550, 0.0],
                                            [1288.210205, 33.304169, 0.0]]),
                        'mask3': np.array([[1288.173828, 49.396580, 0.0], 
                                            [1311.551392, 49.298294, 0.0],
                                            [1311.529663, 55.166191, 0.0],
                                            [1288.167969, 55.420048, 0.0]]),
                        'mask4': np.array([[940.801270, 24.622738, 0.0], 
                                            [961.428162, 24.701952, 0.0],
                                            [961.460388, 36.956326, 0.0],
                                            [940.611572, 37.158779, 0.0]]),
                        'mask5': np.array([[940.572083, 49.993092, 0.0], 
                                            [961.221802, 49.856434, 0.0],
                                            [961.204834, 62.050415, 0.0],
                                            [940.693848, 62.294289, 0.0]])}
        self.pose = None
        self.img = None


    def state_callback(self, msg):
        pos = msg.pose[1].position
        self.pose = [pos.x, pos.y, pos.z]

    def image_callback(self, img_msg):
        # Log some info about the image topic
        # rospy.loginfo(img_msg.header)
        cur_pos = self.pose
        self.count += 1
        # if self.count % 10 != 0:
        #     return
        # self.pic_index += 1

        # Try to convert the ROS image to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # img = PIL.Image.fromarray(cv_image)
            self.img = cv_image
        #     hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # self.show_image(cv_image)
        #     # cv2.waitKey(100)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        

        # # Key points detection using trained nn.
        # keypoints_mask1 = self.detect_keypoints(self.nets[0], img, self.device)
        # keypoints_mask2 = self.detect_keypoints(self.nets[1], img, self.device)
        # keypoints_mask3 = self.detect_keypoints(self.nets[2], img, self.device)
        # keypoints_mask4 = self.detect_keypoints(self.nets[3], img, self.device)
        # keypoints_mask5 = self.detect_keypoints(self.nets[4], img, self.device)

        # # # Pose estimation using PnP
        # keypoints_list = [keypoints_mask1, keypoints_mask2, keypoints_mask3, keypoints_mask4, keypoints_mask5]
        # keypoints_detected = []
        # keypoints_ground_truth = []
        # for i in range(5):
        #     for j in range(4):
        #         if keypoints_list[i] is None:
        #             continue
        #         if len(keypoints_list[i][j][0]) == 0:
        #             continue
        #         keypoints_detected.append([keypoints_list[i][j][0][0], keypoints_list[i][j][1][0]])
        #         # print(keypoints_list[i][j])
        #         keypoints_ground_truth.append(self.keypoints['mask{0}'.format(i+1)][j])
        # #     kps.append(kp[i].pt)
        # print(keypoints_detected)
        # self.show_image(cv_image, keypoints_detected)
        # # success, rotation_vector, translation_vector = self.pose_estimation(np.array(keypoints_detected), np.array(keypoints_ground_truth), self.K)
        # # if success:
        # #     # print("Rotation: ", rotation_vector)
        # #     # print("Translation: ", translation_vector)
        # #     # print("")
        # #     Rot = cv2.Rodrigues(rotation_vector)[0]
        # #     # P = np.hstack((Rot, translation_vector))
        # #     # euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
        # #     # euler_angles_degrees = 180 * euler_angles_radians/pi

        # #     camera_position = -np.matrix(Rot).T*np.matrix(translation_vector)
        # #     _, _, _, Qx, Qy, Qz = cv2.RQDecomp3x3(Rot)
        # #     camera_roll = atan2(Qx[2][1], Qx[2][2])
        # #     camera_pitch = atan2(-Qy[2][0], sqrt(Qy[2][1]**2 + Qy[2][2]**2))
        # #     camera_yaw = atan2(Qz[1][0], Qz[0][0])
        # #     # camera_roll = atan2(-Rot[2][1], Rot[2][2])
        # #     # camera_pitch = asin(Rot[2][0])
        # #     # camera_yaw = atan2(-Rot[1][0], Rot[0][0])
        # #     print("Camera position: ", camera_position)
        # #     print("")
        # #     print("Camera orientation: ", camera_roll, camera_pitch, camera_yaw)
        # #     print("")
        # #     # print("Aircraft position: ", cur_pos)
        # #     # print("")
        # # else:
        #     print("Pose Estimation Failed.")

    def show_image(self, img, keypoints):
        # kp_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
        kp_img = img
        for i in range(len(keypoints)):
            kp_img = cv2.circle(kp_img, keypoints[i], radius=1, color=(0, 0, 255))
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(3)

    # def predict_img(self, net, full_img, device, scale_factor=0.5, out_threshold=0.5):
    #     net.eval()
    #     img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    #     img = img.unsqueeze(0)
    #     img = img.to(device=device, dtype=torch.float32)

    #     with torch.no_grad():
    #         output = net(img).cpu()
    #         output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    #         if net.n_classes > 1:
    #             mask = output.argmax(dim=1)
    #         else:
    #             mask = torch.sigmoid(output) > out_threshold

    #     return mask[0].long().squeeze().numpy()


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

    def find_largest_polytope(self, v):
        v1 = 0
        v2 = 1
        v3 = 2
        v4 = 3
        largest_area = -1
        for i in range(v.shape[0]-3):
            for j in range(i+1,v.shape[0]-2):
                for k in range(j+1, v.shape[0]-1):
                    for l in range(k+1, v.shape[0]):
                        area = 1/2*((v[i,0]*v[j,1]+v[j,0]*v[k,1]+v[k,0]*v[l,1]+v[l,0]*v[i,1])-\
                            (v[j,0]*v[i,1]+v[k,0]*v[j,1]+v[l,0]*v[k,1]+v[i,0]*v[l,1]))
                        if area > largest_area:
                            largest_area = area 
                            v1 = i 
                            v2 = j 
                            v3 = k
                            v4 = l        
        return v[[v1, v2, v3, v4], :]
    
    # def detect_keypoints(self, net, img, device):
    #     mask = self.predict_img(net=net, full_img=img, device=device)

    #     pixels = np.where(mask>0)
    #     pixels_array = np.vstack((pixels[1],pixels[0])).T

    #     if pixels_array.size == 0:
    #         return None
        
    #     hull = scipy.spatial.ConvexHull(pixels_array)
    #     hull_vertices = pixels_array[hull.vertices,:]
    #     center_point = np.mean(hull_vertices,axis=0)
    #     # plt.plot(center_point[0], center_point[1], 'b*')

    #     pts = self.find_largest_polytope(hull_vertices)
        
    #     upleft = np.where((pts[:,0]<center_point[0]) & (pts[:,1]<center_point[1]))
    #     upleft_vertices = pts[upleft[0],:]
    #     upleft_vertex = [upleft_vertices[:,0],upleft_vertices[:,1]]
    #     # plt.plot(upleft_vertices[:,0],upleft_vertices[:,1],'r*')

    #     bottomleft = np.where((pts[:,0]<center_point[0]) & (pts[:,1]>center_point[1]))
    #     bottomleft_vertices = pts[bottomleft[0],:]
    #     bottomleft_vertex = [bottomleft_vertices[:,0],bottomleft_vertices[:,1]]
    #     # plt.plot(bottomleft_vertices[:,0],bottomleft_vertices[:,1],'g*')

    #     bottomright = np.where((pts[:,0]>center_point[0]) & (pts[:,1]>center_point[1]))
    #     bottomright_vertices = pts[bottomright[0],:]
    #     bottomright_vertex = [bottomright_vertices[:,0],bottomright_vertices[:,1]]
    #     # plt.plot(bottomright_vertices[:,0],bottomright_vertices[:,1],'b*')

    #     upright = np.where((pts[:,0]>center_point[0]) & (pts[:,1]<center_point[1]))
    #     upright_vertices = pts[upright[0],:]
    #     upright_vertex = [upright_vertices[:,0],upright_vertices[:,1]]
    #     # plt.plot(upright_vertices[:,0],upright_vertices[:,1],'y*')
    #     return [upleft_vertex, bottomleft_vertex, bottomright_vertex, upright_vertex]

    def pose_estimation(self, object_points, image_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        return cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

class Aircraft():
    def __init__(self, ctrlArgs):
        # self.airspeed = ctrlArgs[0] # Air speed velocity of the aircraft
        self.gravity = 9.81 # Acceleration due to gravity

        self.path = ctrlArgs[0] # Predicted path that the agent will be following over the time horizon
        # path function should be of the form path(time, pathArgs), and will return the reference state of the form ([x,y,z],[v_xy, v_z, heading_rate]) at the time specified

        self.K1 = ctrlArgs[1] # Control gains for getting reference velocities

        self.K2 = ctrlArgs[2] # Control gains for getting inputs to hovercraft

        self.time = ctrlArgs[3] # Current time

        self.pathArgs = ctrlArgs[4] # Arguments needed for the path that the aircraft is tracking

        self.ctrlType = ctrlArgs[5] # What kind of controller is the aircraft applying

    def aircraft_dynamics(self, state, t, control_type):
        # This function are the "tracking" dynamics used for the dubin's aircraft
        heading, pitch, velocity = state[3:]

        heading = heading%(2*pi)
        if heading > pi:
            heading = heading - 2*pi
        pitch = pitch%(2*pi)
        if pitch > pi:
            pitch = pitch - 2*pi
        if velocity > 10:
            velocity = 10

        # Here we get the reference "state" ([x, y, z, heading]) and the reference "input" ([v_xy_ref, v_z_ref, heading_rate_ref])
        ref_state, ref_input = self.path(self.time + t, self.pathArgs)
        # print(ref_state, ref_input)
        # Now we use a PID controller to get the actual inputs to the aircraft dynamics
        heading_rate, pitch_rate, acceleration = self.aircraft_tracking_control(state, ref_state, ref_input)

        # Time derivative of the states
        dxdt = velocity*cos(heading)*cos(pitch)
        dydt = velocity*sin(heading)*cos(pitch)
        dzdt = velocity*sin(pitch)
        dheadingdt = heading_rate
        dpitchdt = pitch_rate
        dveldt = acceleration

        # accel_max = 10
        accel_max = 20
        # accel_max = 100
        heading_rate_max = pi/2
        # heading_rate_max = pi/3
        pitch_rate_max = pi/2
        # pitch_rate_max = pi/3
        if abs(dveldt)>accel_max:
            dveldt = np.sign(dveldt)*accel_max
        if abs(dpitchdt)>pitch_rate_max*1:
            dpitchdt = np.sign(dpitchdt)*pitch_rate_max
        if abs(dheadingdt)>heading_rate_max:
            dheadingdt = np.sign(dheadingdt)*heading_rate_max

        # print(acceleration)
        return [dxdt, dydt, dzdt, dheadingdt, dpitchdt, dveldt]
        
    def aircraft_tracking_control(self, state, ref_state, ref_input):
        # This is a PID controller used to track some trajectory
        x, y, z, heading, pitch, velocity = state
        v_rf, pitch_rf, heading_rate_rf = self.hover_tracking_control([x, y, z, heading], ref_state, ref_input)

        heading_rate = heading_rate_rf
        pitch_rate = self.K2[0]*(pitch_rf - pitch)
        acceleration = np.sign(v_rf - velocity)*min(abs(self.K2[1]*(v_rf - velocity)),100)
        acceleration = self.K2[1]*(v_rf - velocity)

        return [heading_rate, pitch_rate, acceleration]

    def hover_tracking_control(self, state, ref_state, ref_input):
        # This is a Lyapunov based feedback controller for the kinematic vehicle (which we call "hover" vehicle)
        # This is taken from the FACTEST work
        x, y, z, heading = state
        x_rf, y_rf, z_rf, heading_rf = ref_state
        v_xy_rf, v_z_rf, heading_rate_rf = ref_input

        err_x = cos(heading)*(x_rf - x) + sin(heading)*(y_rf - y)
        err_y = -sin(heading)*(x_rf - x) + cos(heading)*(y_rf - y)
        err_z = z_rf - z
        err_heading = heading_rf - heading

        v_xy = v_xy_rf*cos(err_heading) + self.K1[0]*err_x
        v_z = v_z_rf + self.K1[3]*err_z
        heading_rate = heading_rate_rf + v_xy_rf*(self.K1[1]*err_y + self.K1[2]*sin(err_heading))

        v = sqrt(v_xy**2 + v_z**2)
        # pitch = atan2(v_z, v_xy)
        # Approach the runway with 3 degree slope.
        pitch = np.deg2rad(-3)

        input_signal = [v, pitch, heading_rate]

        return input_signal

    def update_state(self, state, time_step, time_bound, control_type = 'tracking'):
        sol = odeint(self.aircraft_dynamics, state, np.arange(0, time_bound, time_step), args = (control_type,))
        ref_state, ref_input = self.path(self.time + time_bound, self.pathArgs)
        # print('goal_state', ref_state, ref_input)
        return sol

    def TC_simulate(self, initialCondition, time_horizon, time_step) -> np.ndarray:
        '''Inputs for TC_simulate function are as follows:
            mode: mode that the system is operating in (ONLY NORMAL MODE FOR OUR CAR EXAMPLE)
            initialCondition (list): initial condition for simulation
            time_bound (float): simulation time
            time_step (float): time step between time stamps
            LEAVE lane_map AS NONE
        Output for TC_simulate function:
            trace (np.ndarray): Simulation trace of the form np.ndarray([[t0, state0],
                                                                         [t1, state1],
                                                                              ...    ])
        '''

        trace = []
        state = initialCondition

        new_states = self.update_state(state, time_step, time_horizon, control_type = self.ctrlType)

        for i, new_state in enumerate(new_states):
            trace.append([i * time_step] + list(new_state))

        return np.array(trace)

# For the default map
# def path(t, pathArgs, x = 1653.639038, y = -543.820251, z = 55.7, yaw = -0.230609+pi):
#     k = np.tan(3*(pi/180))
#     v = 0.1
#     # return [x - 0.001*t, y, z - 0.001*t, yaw], [-0.001, -0.001, 0]
#     return [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, z - k*v*t, yaw], [-v, -k*v, 0]
#     # return [x, y, z, yaw], [0, 0, 0]

def path(t, pathArgs, x = 2049.805183026255, y = 37.79750966976688, z = 125.96311171321648, yaw = -0.0063+pi):
    k = np.tan(3*(pi/180))
    v = 0.2
    # return [x - 0.001*t, y, z - 0.001*t, yaw], [-0.001, -0.001, 0]
    if  z - k*v*t <= 0:
        return  [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, 0, yaw], [-v, 0.0, 0]
    return [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, z - k*v*t, yaw], [-v, -k*v, 0]

# def path(t, pathArgs, x = 1200.0, y = 39.470692, z = 50.0, yaw = -0.0063+pi):
#     k = np.tan(3*(pi/180))
#     v = 0.2
#     # return [x - 0.001*t, y, z - 0.001*t, yaw], [-0.001, -0.001, 0]
#     return [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, z - k*v*t, yaw], [-v, -k*v, 0]

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

def update_aircraft_position():
    # Predicted path that the agent will be following over the time horizon

    def path(t, pathArgs, x = -1500, y = 0, z = 125.96311171321648, yaw = 0):
        k = np.tan(3*(pi/180))
        v = 0.2
        # return [x - 0.001*t, y, z - 0.001*t, yaw], [-0.001, -0.001, 0]
        if  z - k*v*t <= 0:
            return  [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, 0, yaw], [-v, 0.0, 0]
        return [cos(yaw)*(v*t) + x, sin(yaw)*(v*t) + y, z - k*v*t, yaw], [-v, -k*v, 0]

    testPath = path 
    K1 =  [10, 10, 1, 1] # Control gains for getting reference velocities
    K2 = [1, 10]
    pathArgs = None

    controlArgs = (testPath, K1, K2, 30, pathArgs, 'tracking')
    # The relative orientation of the runway with respect to the world frame is given by yaw = -0.230609+pi radians
    # initial_state = [1653.639038, -543.820251, 55.7, -0.230609+pi, 0, 0]
    # initial_state = [1649.813721, 40.317493, 55.0, -0.0063+pi, 0, 0]
    # initial_state = [1200.0, 39.470692, 50.0, -0.0063+pi, 0, 0]
    initial_state = [-1500, 0, 125.96311171321648, 0, 0, 0]

    agent = Aircraft(controlArgs)
    perception = Perception()
    init_msg = create_state_msd(initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4], initial_state[5])
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(init_msg)
    except rospy.ServiceException:
        print("Service call failed")

    # cur_state = [1653.639038, -543.820251, 55.7, -0.230609+pi, 0, 0]
    # cur_state = [1649.813721, 40.317493, 55.0, -0.0063+pi, 0, 0]
    # cur_state = [1200.0, 39.470692, 50.0, -0.0063+pi, 0, 0]
    cur_state = initial_state

    idx = 0
    while not rospy.is_shutdown():
        cur_trace = agent.TC_simulate(cur_state, 0.011, 0.01)
        # print(cur_trace)
        agent.time += 0.011
        # print(cur_trace)
        cur_state = cur_trace[-1][1:]
        # print(cur_state)
        state_msg = create_state_msd(cur_state[0], cur_state[1], cur_state[2], 0, cur_state[4], cur_state[3])

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException:
            print("Service call failed")

        if perception.img is None:
            continue

        time.sleep(0.01)
        data_fn = os.path.join(data_path, f"data.txt")
        q = squaternion.Quaternion.from_euler(0, cur_state[4], cur_state[3])
        with open(data_fn,'a+') as f:
            f.write(f"\n{idx},{cur_state[0]},{cur_state[1]},{cur_state[2]},{q.x},{q.y},{q.z},{q.w}")

        cv2.imshow('camera',perception.img)
        cv2.waitKey(3)
        path = os.path.join(img_path, f"img_{idx}.png")
        if perception.img is not None:
            cv2.imwrite(path, perception.img)
        idx += 1


if __name__ == "__main__":
    rospy.init_node('update_poses', anonymous=True)
    
    try:
        update_aircraft_position()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
        
