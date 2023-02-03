/*
 * Copyright 2015 James Jackson MAGICC Lab, BYU, Provo, UT
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

#include "magicc_sim_plugins/autolevel_plugin.h"
#include "magicc_sim_plugins/gz_compat.h"

namespace gazebo {

AutoLevelPlugin::AutoLevelPlugin() : ModelPlugin() {}

AutoLevelPlugin::~AutoLevelPlugin() {
  GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(updateConnection_);
}

void AutoLevelPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  // Configure Gazebo Integration
  model_ = _model;
  world_ = model_->GetWorld();
  namespace_.clear();

  // Load SDF parameters
  if (_sdf->HasElement("namespace")) {
    namespace_ = _sdf->GetElement("namespace")->Get<std::string>();
  } else {
    gzerr << "[AutoLevelPlugin] Please specify a namespace";
  }

  if (_sdf->HasElement("sensorLink")) {
    sensor_link = model_->GetLink(_sdf->GetElement("sensorLink")->Get<std::string>());
  } else{
    gzerr << "[AutoLevelPlugin] Please specify a sensor link";
  }

  std::string link_name;
  if (_sdf->HasElement("modelLink")){
    model_link = model_->GetLink(_sdf->GetElement("modelLink")->Get<std::string>());
  }else{
    gzerr << "[AutoLevelPlugin] Please specify a linkName of the forces and moments plugin.\n";
  }



  // Connect Gazebo Update
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&AutoLevelPlugin::OnUpdate, this, _1));


  // Set the axes of the gimbal
//  yaw_joint_->SetAxis(0, math::Vector3(0, 0, 1));
//  pitch_joint_->SetAxis(0, math::Vector3(0, 1, 0));
//  roll_joint_->SetAxis(0, math::Vector3(1, 0, 0));

  // Initialize
  this->Reset();
}

void AutoLevelPlugin::Reset()
{

}

// Return the Sign of the argument
void AutoLevelPlugin::OnUpdate(const common::UpdateInfo & _info)
{
    GazeboVector global_pose = GZ_COMPAT_GET_EULER(GZ_COMPAT_GET_ROT(GZ_COMPAT_GET_WORLD_COG_POSE(model_link)));
    GazeboVector relative_pose = GZ_COMPAT_GET_EULER(GZ_COMPAT_GET_ROT(GZ_COMPAT_GET_RELATIVE_POSE(model_link)));

    double roll = -GZ_COMPAT_GET_X(global_pose);
    double pitch =  -GZ_COMPAT_GET_Y(global_pose);
    double yaw = -GZ_COMPAT_GET_Z(relative_pose);

    sensor_link->SetRelativePose(GazeboPose(0, 0, -.50, roll, pitch, yaw));

}

GZ_REGISTER_MODEL_PLUGIN(AutoLevelPlugin);

} // namespace
