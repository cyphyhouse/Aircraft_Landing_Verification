/*
 * Copyright 2016 James Jackson MAGICC Lab - BYU - Provo, UT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef cu_sim_PLUGINS_GIMBAL_PLUGIN_H
#define magicc_sim_PLUGINS_GIMBAL_PLUGIN_H


//#include <gazebo/math/gzmath.hh>
//#include <ctime>
//#include <stdio.h>

#include <gazebo/common/common.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include "geometry_msgs/Vector3Stamped.h"
#include <magicc_sim_plugins/common.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdio.h>

#include <boost/bind.hpp>

#define PI 3.141592

namespace gazebo {


class GimbalPlugin : public ModelPlugin {
public:
  GimbalPlugin();
  ~GimbalPlugin();
  void commandCallback(const geometry_msgs::Vector3StampedConstPtr &msg);

protected:

  void Reset();
  virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
  void OnUpdate(const common::UpdateInfo & _info);
  int sign(double x);

private:
  // ROS variables
  ros::NodeHandle* nh_;
  ros::Subscriber command_sub_;
  ros::Publisher pose_pub_;

  // Pointer to the gazebo items.
  physics::LinkPtr link_;
  physics::JointControllerPtr joint_controller_;
  physics::JointPtr yaw_joint_;
  physics::JointPtr roll_joint_;
  physics::JointPtr pitch_joint_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  std::string namespace_;


  // Pointer to the update event connection
  event::ConnectionPtr updateConnection_;

  // Time
  double previous_time_;
  double current_time_;

  // Commands
  double yaw_desired_;
  double pitch_desired_;
  double roll_desired_;
  double yaw_actual_;
  double pitch_actual_;
  double roll_actual_;

  // Flags
  bool use_slipring_;
  bool auto_stabilize_;

  // Filters on Axes
  std::unique_ptr<FirstOrderFilter<double>> yaw_filter_;
  std::unique_ptr<FirstOrderFilter<double>> pitch_filter_;
  std::unique_ptr<FirstOrderFilter<double>> roll_filter_;

  double time_constant_;
};
} // namespace gazebo
#endif //magicc_sim_PLUGINS_GIMBAL_PLUGIN_H
