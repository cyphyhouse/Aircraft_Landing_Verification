#!/usr/bin/env python

"""!@brief PositionChecker module to calculate the lights currently present in the robots FoV.

@details Subscribes to the robot's ground truth position data and calculates its distance to every light in the
simulation. Lights currently in the FoV can be passed to other modules with via callbacks.

@file position_checker.py PositionChecker class.

@author Martin Schuck

@date 19.09.2020
"""

import numpy as np
from numba import jit

import rospy
from nav_msgs.msg import Odometry


class PositionChecker:
    """!@brief Contains all logic for distance calculation and index selection.

    @details The PositionChecker is triggered by the robot's ground truth position data messages. Therefore, the
    frequency of the PositionChecker is dependent on the topic frequency!
    """

    def __init__(self, light_positions, odom_topic, *, distance_threshold=100, max_lights=20, callbacks=[]):
        """!@brief PositionChecker constructor.

        @details Initializes the subscriber and callbacks. Also compiles the numba accelerated index selection function.

        @param light_positions Array of light positions. Array should be of dimensions np.ndarray(x,2). The rows are x
        and y coordinate of the lights respectively.
        @param odom_topic The ROS topic that the robot's odometry is published at.
        @param distance_threshold Maximum light rendering distance in meters. Default 100.
        @param max_lights Maximum amount of lights chosen at the same time.
        @param callbacks List of callbacks to execute when finishing an index selection.

        @note Numba compilation might take some time. For more information, see @ref __check_lights_boost.
        """
        self.light_positions = light_positions
        self.distance_threshold = distance_threshold
        self.max_lights = max_lights
        self.start_callback = False
        self.ground_truth_subscriber = rospy.Subscriber(odom_topic, Odometry, self._odometry_callback)
        self.callbacks = callbacks
        self.__init_numba_jit()

    def start(self):
        """!@brief Starts the PositionChecker.

        @details Sets the ground truth position callback lock flag. Callback is locked during initialization to avoid
        startup errors.
        """
        self.start_callback = True
        rospy.loginfo("Started PositionChecker.")

    def _odometry_callback(self, odom):
        """!@brief Callback for the ground truth position subscriber.

        @details Acts as a wrapper to @ref __check_lights_boost. Extracts the position from the odometry data and calls
        the compiled function. Executes all callbacks with the array of active lights indices on completion.

        @param odom Ground truth odometry data from the subscriber.
        """
        if self.start_callback:
            position = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
            active_lights = self.__check_lights_boost(self.light_positions, position, self.distance_threshold,
                                                      self.max_lights)
            active_lights = active_lights[active_lights >= 0]
            for callback in self.callbacks:
                callback(active_lights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def __check_lights_boost(light_positions, position, distance_threshold, max_lights):
        """!@brief Numba accelerated indices selection function.

        @details Calculates the distance from the robot to all lights. If more than max_lights are found to be within
        the threshold, orders the lights and selects the max_lights closest ones. If not, selects all lights within the
        threshold.

        @param light_positions Array of light positions. Array should be of dimensions np.ndarray(x,2). The rows are x
        and y coordinate of the lights respectively.
        @param position Current robot position. Should be of type np.ndarray(2,).
        @param distance_threshold Maximum light rendering distance in meters.
        @param max_lights Maximum amount of lights chosen at the same time.

        @return The indices of lights within the FoV, restricted by the maximum number of lights. If less than
        max_lights are found, array contains -1 integers for unused array slots. Required because numba works best with
        constant return size.
        """
        light_distance = np.sqrt(np.sum((light_positions - position)**2, axis=1))
        if np.sum(light_distance <= distance_threshold) >= max_lights:
            load_index = np.argsort(light_distance)[:max_lights].flatten()  # np.argpartition not supported by numba.
        else:
            tmp = np.argwhere(light_distance <= distance_threshold).flatten()
            load_index = np.ones(max_lights, dtype=np.int64) * -1  # Mark invalid indices.
            load_index[0:tmp.shape[0]] = tmp
        return load_index

    def __init_numba_jit(self):
        """!@brief Numba compile on object initialization function.

        @details Calls the @ref __check_lights_boost function once with dummy values to trigger numba jit compilation.
        Numba jit instead of ahead of time compilation was chosen to enable CPU architecture specific optimizations.
        """

        rospy.loginfo('Compiling numba functions for PositionChecker.')
        init_light_pos = np.zeros((10, 2))
        init_pos = np.ones(2)
        self.__check_lights_boost(init_light_pos, init_pos, self.distance_threshold, self.max_lights)
        rospy.loginfo('Numba compilation completed.')
