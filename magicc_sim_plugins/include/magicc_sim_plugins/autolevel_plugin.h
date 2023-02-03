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

#ifndef magicc_sim_PLUGINS_AUTOLEVEL_PLUGIN_H
#define magicc_sim_PLUGINS_AUTOLEVEL_PLUGIN_H


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


class AutoLevelPlugin : public ModelPlugin {
public:
  AutoLevelPlugin();
  ~AutoLevelPlugin();

protected:

  void Reset();
  virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
  void OnUpdate(const common::UpdateInfo & _info);

private:
  // Pointer to the gazebo items.
  physics::LinkPtr sensor_link;
  physics::LinkPtr model_link;
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  std::string namespace_;

  // Pointer to the update event connection
  event::ConnectionPtr updateConnection_;

};
} // namespace gazebo

#endif
