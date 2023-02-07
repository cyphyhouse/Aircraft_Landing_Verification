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

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from rosplane_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from sensor_msgs.msg import Image

import os
img_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/imgs'
class Perception():
    def __init__(self, name=''):
        self.name = name

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()
        # cv2.namedWindow("Image Window", 1)

        self.count = -1

        # Camera matrix
        self.K = np.array([[205.46963709898583, 0.0, 320.5],
                    [0.0, 205.46963709898583, 240.5],
                    [0.0, 0.0, 1.0]])
        
        # 3D locations of the six markers
        self.marker_points = np.array([
                                        (-124.19, 18.33, 0), # red marker
                                        (-125.06, 12.72, 0), # green marker
                                        (-125.29, 6.26, 0), # blue marker
                                        (-91.10, 11.20, 0), # purple marker
                                        (-93.17, -2.39, 0), # cyan marker
                                        (-92.11, 4.16, 0), # yellow marker
                                        # (-134.77, 27.14, 0), # light red marker
                                        # (-134.04, 12.08, 0), # light blue marker
                                        # (-139.79, 2.48, 0) # light green marker
                                        (-106.38, -12.04, 0), # orange marker
                                        (-102.21, 25.83, 0) # white marker
                                    ])
        self.pose = None

    def state_callback(self, msg):
        pos = msg.pose[1].position
        self.pose = [pos.x, pos.y, pos.z]

    def image_callback(self, img_msg):
        # Log some info about the image topic
        # rospy.loginfo(img_msg.header)
        cur_pos = self.pose
        self.count += 1
        if self.count % 100 != 0:
            return
        # Try to convert the ROS image to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # # Find the red marker
            # mask_red_1 = cv2.inRange(hsv, (0, 200, 20), (5, 255, 255))
            # mask_red_2 = cv2.inRange(hsv, (175, 200, 20), (180 ,255, 255))

            # mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
            # img_red = cv2.bitwise_and(cv_image, cv_image, mask=mask_red)
            # red_image_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)

            # # Find the green marker
            # mask_green = cv2.inRange(hsv, np.array([50, 200, 100]), np.array([70, 255, 255]))
            # img_green = cv2.bitwise_and(cv_image, cv_image, mask=mask_green)
            # green_image_gray = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)

            # # Find the blue marker
            # mask_blue = cv2.inRange(hsv, np.array([100, 200, 0]), np.array([140, 255, 255]))
            # img_blue = cv2.bitwise_and(cv_image, cv_image, mask=mask_blue)
            # blue_image_gray = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)

            # # Find the purple marker
            # mask_purple = cv2.inRange(hsv, np.array([140, 200, 70]), np.array([160, 250, 250]))
            # img_purple = cv2.bitwise_and(cv_image, cv_image, mask=mask_purple)
            # purple_image_gray = cv2.cvtColor(img_purple, cv2.COLOR_BGR2GRAY)

            # # Find the cyan marker
            # mask_cyan = cv2.inRange(hsv, np.array([80, 200, 70]), np.array([90, 250, 250]))
            # img_cyan = cv2.bitwise_and(cv_image, cv_image, mask=mask_cyan)
            # cyan_image_gray = cv2.cvtColor(img_cyan, cv2.COLOR_BGR2GRAY)

            # # Find the yellow marker
            # mask_yellow = cv2.inRange(hsv, np.array([20, 200, 100]), np.array([30, 255, 255]))
            # img_yellow = cv2.bitwise_and(cv_image, cv_image, mask=mask_yellow)
            # yellow_image_gray = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
            # Find the red marker
            mask_red_1 = cv2.inRange(hsv, (0, 50, 20), (5, 255, 255))
            mask_red_2 = cv2.inRange(hsv, (175, 50, 20), (180 ,255, 255))

            mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
            img_red = cv2.bitwise_and(cv_image, cv_image, mask=mask_red)
            red_image_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)

            # Find the green marker
            mask_green = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 255, 255]))
            img_green = cv2.bitwise_and(cv_image, cv_image, mask=mask_green)
            green_image_gray = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)

            # Find the blue marker
            mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
            img_blue = cv2.bitwise_and(cv_image, cv_image, mask=mask_blue)
            blue_image_gray = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)

            # Find the purple marker
            mask_purple = cv2.inRange(hsv, np.array([140, 50, 70]), np.array([160, 250, 250]))
            img_purple = cv2.bitwise_and(cv_image, cv_image, mask=mask_purple)
            purple_image_gray = cv2.cvtColor(img_purple, cv2.COLOR_BGR2GRAY)

            # Find the cyan marker
            mask_cyan = cv2.inRange(hsv, np.array([80, 50, 70]), np.array([90, 250, 250]))
            img_cyan = cv2.bitwise_and(cv_image, cv_image, mask=mask_cyan)
            cyan_image_gray = cv2.cvtColor(img_cyan, cv2.COLOR_BGR2GRAY)

            # Find the yellow marker
            mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            img_yellow = cv2.bitwise_and(cv_image, cv_image, mask=mask_yellow)
            yellow_image_gray = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
            # Find the orange marker
            mask_orange = cv2.inRange(hsv, np.array([10, 200, 100]), np.array([19, 255, 255]))
            img_orange = cv2.bitwise_and(cv_image, cv_image, mask=mask_orange)
            orange_image_gray = cv2.cvtColor(img_orange, cv2.COLOR_BGR2GRAY)

            # Find the white marker
            mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([0, 255, 255]))
            img_white = cv2.bitwise_and(cv_image, cv_image, mask=mask_white)
            white_image_gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
            # # Find the light red marker
            # mask_red_light_1 = cv2.inRange(hsv, (0, 50, 20), (5, 150, 255))
            # mask_red_light_2 = cv2.inRange(hsv, (175, 50, 20), (180 ,150, 255))

            # mask_red_light = cv2.bitwise_or(mask_red_light_1, mask_red_light_2)
            # # mask_red_light = cv2.inRange(hsv, np.array([140, 50, 70]), np.array([160, 150, 250]))
            # img_red_light = cv2.bitwise_and(cv_image, cv_image, mask=mask_red_light)
            # light_red_image_gray = cv2.cvtColor(img_red_light, cv2.COLOR_BGR2GRAY)

            # # Find the light blue marker
            # mask_blue_light = cv2.inRange(hsv, np.array([80, 50, 70]), np.array([90, 150, 250]))
            # img_blue_light = cv2.bitwise_and(cv_image, cv_image, mask=mask_blue_light)
            # light_blue_image_gray = cv2.cvtColor(img_blue_light, cv2.COLOR_BGR2GRAY)

            # # Find the light green marker
            # mask_green_light = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 150, 255]))
            # img_green_light = cv2.bitwise_and(cv_image, cv_image, mask=mask_green_light)
            # light_green_image_gray = cv2.cvtColor(img_green_light, cv2.COLOR_BGR2GRAY)

            # print(cv_image)
            kp, absent_markers = self.get_feature_points(np.array([red_image_gray, green_image_gray, blue_image_gray, 
                                                        purple_image_gray, cyan_image_gray, yellow_image_gray, 
                                                        orange_image_gray, white_image_gray]))
            kp_img = cv2.drawKeypoints(cv_image, kp, None, color=(0, 255, 0), flags=0)

            cv2.imwrite(os.path.join(img_path, "img_{0}.jpg".format(self.count)), kp_img)
            # cv2.waitKey(100)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        
        # Pose estimation using PnP
        obj_pts = []
        # print(kp)
        kps = []
        for i in range(8):
            if absent_markers[i] == 0:
                obj_pts.append(self.marker_points[i])
        for i in range(len(kp)):
            # print(kp[i].pt)
            kps.append(kp[i].pt)

        success, rotation_vector, translation_vector = self.pose_estimation(np.array(obj_pts), np.array(kps), self.K)
        if success:
            # print("Rotation: ", rotation_vector)
            # print("Translation: ", translation_vector)
            # print("")
            Rot = cv2.Rodrigues(rotation_vector)[0]
            # P = np.hstack((Rot, translation_vector))
            # euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
            # euler_angles_degrees = 180 * euler_angles_radians/pi

            camera_position = -np.matrix(Rot).T*np.matrix(translation_vector)
            _, _, _, Qx, Qy, Qz = cv2.RQDecomp3x3(Rot)
            camera_roll = atan2(Qx[2][1], Qx[2][2])
            camera_pitch = atan2(-Qy[2][0], sqrt(Qy[2][1]**2 + Qy[2][2]**2))
            camera_yaw = atan2(Qz[1][0], Qz[0][0])
            # camera_roll = atan2(-Rot[2][1], Rot[2][2])
            # camera_pitch = asin(Rot[2][0])
            # camera_yaw = atan2(-Rot[1][0], Rot[0][0])
            print("Camera position: ", camera_position)
            print("Camera orientation: ", camera_roll, camera_pitch, camera_yaw)
            print("")
            print("Aircraft position: ", cur_pos)
            print("")
        else:
            print("Pose Estimation Failed.")

    def show_image(self, img):
        cv2.imshow("Image Window", img)
        # cv2.waitKey(3)

    def get_feature_points(self, imgs):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and compute the descriptors with ORB
        kp_red = orb.detect(imgs[0])
        kp_green = orb.detect(imgs[1])
        kp_blue = orb.detect(imgs[2])
        kp_purple = orb.detect(imgs[3])
        kp_cyan = orb.detect(imgs[4])
        kp_yellow = orb.detect(imgs[5])

        kp_orange = orb.detect(imgs[6])
        kp_white = orb.detect(imgs[7])
        # kp_red_light = orb.detect(imgs[6])
        # kp_blue_light = orb.detect(imgs[7])
        # kp_green_light = orb.detect(imgs[8])
        # print(kp_green)
        all_kps = [kp_red, kp_green, kp_blue, kp_purple, kp_cyan, kp_yellow, kp_orange, kp_white]
        absent_color = np.zeros(len(all_kps))
        kps = []
        for i in range(len(all_kps)):
            if len(all_kps[i]) == 0:
                absent_color[i] = 1
            else:
                kps.append(all_kps[i][0])
        return kps, absent_color

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

        accel_max = 10
        # accel_max = 20
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
        pitch = atan2(v_z, v_xy)

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

def path(t, pathArgs, x = 60, y = -30.0, z = 30.0, yaw = 3.038):
    return [x - 0.001*t, y, z - 0.001*t, yaw], [-0.001, -0.001, 0]
    # return [x, y, z, yaw], [0, 0, 0]


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
    testPath = path # Predicted path that the agent will be following over the time horizon
    K1 =  [10, 10, 1, 1] # Control gains for getting reference velocities
    K2 = [0.01, 10]
    pathArgs = None

    controlArgs = (testPath, K1, K2, 30, pathArgs, 'tracking')
    initial_state = [60.0, -30.0, 30.0, 3.038, 0, 0]

    agent = Aircraft(controlArgs)
    perception = Perception()
    init_msg = create_state_msd(initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4], initial_state[5])
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(init_msg)
    except rospy.ServiceException:
        print("Service call failed")

    cur_state = [60.0, -30.0, 30.0, 3.038, 0, 0]

    while not rospy.is_shutdown():
        cur_trace = agent.TC_simulate(cur_state, 0.011, 0.01)
        agent.time += 0.011
        # print(cur_trace)
        cur_state = cur_trace[-1][1:]

        state_msg = create_state_msd(cur_state[0], cur_state[1], cur_state[2], 0, cur_state[4], cur_state[3])

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException:
            print("Service call failed")

        time.sleep(0.01)


if __name__ == "__main__":
    rospy.init_node('update_poses', anonymous=True)
    try:
        update_aircraft_position()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Stop updating aircraft positions.")
