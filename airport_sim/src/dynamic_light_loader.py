#!/usr/bin/env python

"""!@brief Script to dynamically load lights close to the robot's current position into a simulation.

@details The DynamicLightLoader class gets initialized with an array of known light positions in the simulation. The
numpy array contains a light's 2D position on its first 2 entries, and its color encoded as an integer on the third. By
setting up a callback in the PositionChecker node, the DynamicLightLoader gets notified of the lights that are currently
in the virtual field of view. If too many lights are included in the radius, the ones closest to the robot get chosen.
The module keeps track on the lights currently present in the simulation to reduce the amount of calls. It also deletes
the lights that are no longer in the FoV.

@file dynamic_light_loader.py DynamicLightLoader class and node script.

@author Martin Schuck

@date 19.09.2020
"""

import numpy as np
from pathlib import Path

import yaml
import rospy
from tf.transformations import quaternion_from_euler
from gazebo_msgs.srv import SetLightProperties, GetLightProperties, SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion

from utils import test_light_array  # pylint: disable=import-error
from position_checker import PositionChecker  # pylint: disable=import-error


class DynamicLightLoader:
    """!@brief Contains the logic for dynamically deleting and adding lights into the simulation.

    @details The DynamicLightLoader makes use of the Gazebo SpawnModel and DeleteModel services to add and remove
    lights. Calculation of the closest lights and the callback trigger is outsourced to the PositionChecker module. The
    light models used are included in the airport_sim package. @see position_checker.py
    """

    def __init__(self, light_array):
        """!@brief DynamicLightLoader constructor.

        @details Initializes the ROS node and loads the .sdf model files from the airport_sim package.

        @param light_array Array of light positions and their color. Array should be of dimensions np.ndarray(x,3). The
        first two rows are x and y coordinate of the lights, the third one encodes the color. 0 -> red, 1 -> green,
        2 -> blue.
        """
        rospy.init_node(name='DynamicLightLoader')
        self._gazebo_model_spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self._gazebo_model_delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.package_path = Path(__file__).resolve().parent.parent
        self.light_model = [None] * 3
        self._init_light_model_xml()
        self.light_array = light_array
        self.active_lights = set()
        self._default_orientation = Quaternion(*quaternion_from_euler(0., 0., 0.))
        odom_topic, distance_threshold, max_lights = self._read_config(self.package_path)
        self.position_checker = PositionChecker(light_positions=self.light_array[:, 0:2], odom_topic=odom_topic,
                                                distance_threshold=distance_threshold, max_lights=max_lights,
                                                callbacks=[self._checker_callback])

    def _init_light_model_xml(self):
        """!@brief Loads the model XMLs from their files.

        @details Models can be found at airport_sim/gazebo/models/<color>_light.
        """
        with open(self.package_path.joinpath('gazebo', 'models', 'red_light', 'model.sdf'), 'r') as f:
            self.light_model[0] = f.read()
        with open(self.package_path.joinpath('gazebo', 'models', 'green_light', 'model.sdf'), 'r') as f:
            self.light_model[1] = f.read()
        with open(self.package_path.joinpath('gazebo', 'models', 'blue_light', 'model.sdf'), 'r') as f:
            self.light_model[2] = f.read()

    @staticmethod
    def _read_config(path):
        """!@brief Loads the config for the light loader.

        @details Config file is located at airport_sim/config/dynamic_load_config.yaml. If yaml file can't be scanned or
        is missing, reverting to defaults.

        @return The ground truth odometry topic, the FoV distance threshold and the maximum number of allowed lights.
        """
        config_path = path.joinpath('config', 'dynamic_light_load_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            odom_topic = config['odom_topic']
            distance_threshold = config['distance_threshold']
            max_lights = config['max_lights']
            rospy.loginfo('DynamicLightLoader config read complete.')
        except (FileNotFoundError, yaml.scanner.ScannerError) as e:
            rospy.logwarn('DynamicLightLoader config file missing or malformed!')
            rospy.loginfo('Config not available, reverting to default.')
            odom_topic = "/ground_truth/odom"
            distance_threshold = 100
            max_lights = 20
        return odom_topic, distance_threshold, max_lights

    def start(self):
        """!@brief Starts the dynamic loading by starting the PositionChecker.

        @details Since loading the lights is set up as a callback from the PositionChecker, the DynamicLightLoader only
        gets invoked once the PositionChecker gets activated.
        """
        self.position_checker.start()
        rospy.loginfo("Started DynamicLightLoader.")
        rospy.spin()

    def _checker_callback(self, light_indices):
        """!@brief PositionChecker callback.

        @details Callback is registered on PositionChecker initialization. Callback handles the tracking of active
        lights as well as the spawning/deletion.

        @param light_indices The indices of lights that are currently within the FoV. Indices describe the lights'
        position in the original light_array.

        @warning Since the checker callback works with ROS services, working through all requests can take a significant
        amount of time. This is especially true on startup when all lights have to be loaded at once.
        """
        rospy.loginfo_once("Loading initial lights. This might take a few seconds.")
        for index in light_indices:
            if index not in self.active_lights:
                try:
                    position = Point(self.light_array[index][0], self.light_array[index][1], 0.3)
                    response = self._gazebo_model_spawn_service('light'+str(index),
                                                                self.light_model[int(self.light_array[index][2])], '',
                                                                Pose(position, self._default_orientation), 'world')
                    if response.success:
                        self.active_lights.add(index)
                except rospy.ServiceException as e:
                    rospy.loginfo(f"Light spawn service failed. Error code: {e}")
        for index in self.active_lights - set(light_indices):
            try:
                response = self._gazebo_model_delete_service('light'+str(index))
                if response.success:
                    self.active_lights.remove(index)
            except rospy.ServiceException as e:
                rospy.loginfo(f"Light delete service failed. Error code: {e}")
        rospy.loginfo_once("Finished loading initial lights.")

    def load_lights(self, light_array):
        """!@brief Loads the complete array of lights into the simulation.

        @details Lights are not tracked. load_lights is intended as a correctness check of a new light position array.
        Does not recover from spawn failures. Drops the lights instead.

        @param light_array Array of light positions and their color. Array should be of dimensions np.ndarray(x,3). The
        first two rows are x and y coordinate of the lights, the third one encodes the color. 0 -> red, 1 -> green,
        2 -> blue.

        @note This function is supposed to be used for testing. Combine with @ref delete_lights.

        @warning Loading all lights into the simulation can take up to minutes, depending on the number of lights.
        Simulation performance will drastically decline!
        """
        rospy.loginfo("Loading lights into the simulation.")
        for idx, position in enumerate(light_array):
            pose = Pose(Point(x=position[0], y=position[1], z=0.3), self._default_orientation)  # 0.3 for visibility.
            try:
                self._gazebo_model_spawn_service('light'+str(idx), self.light_model[int(position[2])],
                                                 '', pose, 'world')
            except rospy.ServiceException as e:
                rospy.loginfo(f"Light spawn service failed. Error code: {e}")

    def delete_lights(self, light_array):
        """!@brief Delete the complete array of lights from the simulation.

        @details Does not recover from delete failures. Leaves the lights in the simulation instead.

        @param light_array Array of light positions and their color. Array should be of dimensions np.ndarray(x,3). The
        first two rows are x and y coordinate of the lights, the third one encodes the color. 0 -> red, 1 -> green,
        2 -> blue.

        @note This function is supposed to be used for testing. Combine with @ref load_lights.

        @warning Deleting all lights in the simulation can take up to minutes, depending on the number of lights.
        """
        for idx, _ in enumerate(light_array):
            try:
                self._gazebo_model_delete_service('light'+str(idx))
            except rospy.ServiceException as e:
                rospy.loginfo(f"Light delete service failed. Error code: {e}")

    @property
    def light_array(self):
        """!@brief Light array property.

        @details Necessary because light_arrays have to be registered with the PositionChecker module.

        @return The current light_array attribute.
        """
        return self._light_array

    @light_array.setter
    def light_array(self, value):
        """!@brief Light array property setter.

        @details Necessary because light_arrays have to be registered with the PositionChecker module. Also checks for
        correctness of the input type. Does not check for correct structure within the array!

        @param value Array of light positions and their color. Array should be of dimensions np.ndarray(x,3). The first
        two rows are x and y coordinate of the lights, the third one encodes the color. 0 -> red, 1 -> green, 2 -> blue.
        """
        assert(type(value) == np.ndarray)
        assert(value.shape[1] == 3)
        try:
            self.position_checker.light_positions = value[:, 0:2]
        except AttributeError:
            pass
        self._light_array = value


if __name__ == '__main__':
    dynamic_loader = DynamicLightLoader(test_light_array)
    dynamic_loader.start()
