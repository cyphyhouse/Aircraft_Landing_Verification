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

#include "magicc_sim_plugins/gimbal_plugin.h"
#include "magicc_sim_plugins/gz_compat.h"

namespace gazebo {

GimbalPlugin::GimbalPlugin() : ModelPlugin() {}

GimbalPlugin::~GimbalPlugin() {
  GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(updateConnection_);
  if (nh_) {
    nh_->shutdown();
    delete nh_;
  }
}

void GimbalPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  // Configure Gazebo Integration
  model_ = _model;
  world_ = model_->GetWorld();
  namespace_.clear();

  std::string command_topic, pose_topic;

  // Load SDF parameters
  if (_sdf->HasElement("namespace")) {
    namespace_ = _sdf->GetElement("namespace")->Get<std::string>();
  } else {
    gzerr << "[GimbalPlugin] Please specify a namespace";
  }

  if (_sdf->HasElement("commandTopic")) {
    command_topic = _sdf->GetElement("commandTopic")->Get<std::string>();
  } else {
    gzerr << "[GimbalPlugin] Please specify a commandTopic";
  }

  if (_sdf->HasElement("poseTopic")) {
    pose_topic = _sdf->GetElement("poseTopic")->Get<std::string>();
  } else {
    gzerr << "[GimbalPlugin] Please specify a poseTopic";
  }

  if (_sdf->HasElement("yawJoint")) {
    std::string yaw_joint_name = _sdf->GetElement("yawJoint")->Get<std::string>();
    yaw_joint_ = model_->GetJoint(yaw_joint_name);
  } else{
    gzerr << "[GimbalPlugin] Please specify a yawJoint";
  }

  if (_sdf->HasElement("pitchJoint")) {
    std::string pitch_joint_name = _sdf->GetElement("pitchJoint")->Get<std::string>();
    pitch_joint_ = model_->GetJoint(pitch_joint_name);
  } else{
    gzerr << "[GimbalPlugin] Please specify a pitchJoint";
  }

  if (_sdf->HasElement("rollJoint")) {
    std::string roll_joint_name = _sdf->GetElement("rollJoint")->Get<std::string>();
    roll_joint_ = model_->GetJoint(roll_joint_name);
  } else{
    gzerr << "[GimbalPlugin] Please specify a rollJoint";
  }

  if (_sdf->HasElement("timeConstant")) {
    time_constant_ = _sdf->GetElement("timeConstant")->Get<double>();
  } else{
    gzerr << "[GimbalPlugin] Please specify a timeConstant";
  }

  if (_sdf->HasElement("useSlipring")) {
    use_slipring_ = _sdf->GetElement("useSlipring")->Get<bool>();
  } else{
    gzerr << "[GimbalPlugin] Please specify whether to use a slipring";
  }

  getSdfParam<bool>(_sdf, "autoStabilize", auto_stabilize_, false);

  // To perform auto stabilization, we need a pointer to the main link
  if(auto_stabilize_)
  {
    std::string link_name;
    if (_sdf->HasElement("linkName"))
      link_name = _sdf->GetElement("linkName")->Get<std::string>();
    else
      gzerr << "[ROSflight_SIL] Please specify a linkName of the forces and moments plugin.\n";
    link_ = model_->GetLink(link_name);
    if (link_ == NULL)
      gzthrow("[ROSflight_SIL] Couldn't find specified link \"" << link_name << "\".");
  }

  // Connect Gazebo Update
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&GimbalPlugin::OnUpdate, this, _1));

  // Connect ROS
  nh_ = new ros::NodeHandle();
  if(!auto_stabilize_)
  {
    command_sub_ = nh_->subscribe(command_topic, 1, &GimbalPlugin::commandCallback, this);
  }
  pose_pub_ = nh_->advertise<geometry_msgs::Vector3Stamped>(pose_topic, 10);

  // Initialize Commands
  yaw_desired_ = 0.0;
  pitch_desired_ = 0.0;
  roll_desired_ = 0.0;

  // Initialize Filtered Values
  yaw_actual_ = 0.0;
  pitch_actual_ = 0.0;
  roll_actual_ = 0.0;

  // Create the first order filters.
  yaw_filter_.reset(new FirstOrderFilter<double>(time_constant_, time_constant_, yaw_actual_));
  pitch_filter_.reset(new FirstOrderFilter<double>(time_constant_, time_constant_, pitch_actual_));
  roll_filter_.reset(new FirstOrderFilter<double>(time_constant_, time_constant_, roll_actual_));

  // Set the max force allowed to set the angle
#if GAZEBO_MAJOR_VERSION > 5
//  pitch_joint_->SetParam("max_force", 0, 10);
//  yaw_joint_->SetParam("max_force", 0, 10);
//  roll_joint_->SetParam("max_force", 0, 10);
#else
  pitch_joint_->SetMaxForce(0, 10);
  yaw_joint_->SetMaxForce(0, 10);
  roll_joint_->SetMaxForce(0, 10);
#endif

  // Set the axes of the gimbal
  yaw_joint_->SetAxis(0, GazeboVector(0, 0, 1));
  pitch_joint_->SetAxis(0, GazeboVector(0, 1, 0));
  roll_joint_->SetAxis(0, GazeboVector(1, 0, 0));

  // Initialize Time
  this->Reset();

}

void GimbalPlugin::Reset()
{
    previous_time_ = 0.0;
}

// Return the Sign of the argument
void GimbalPlugin::OnUpdate(const common::UpdateInfo & _info)
{
  // Update time
  double dt = _info.simTime.Double() - previous_time_;
  previous_time_ = _info.simTime.Double();

  // Perform Control if auto stabilize flag is on
  if(auto_stabilize_)
  {
    GazeboPose W_pose_W_C = GZ_COMPAT_GET_WORLD_COG_POSE(link_);
    GazeboVector euler_angles = GZ_COMPAT_GET_EULER(GZ_COMPAT_GET_ROT(W_pose_W_C));
    double phi = GZ_COMPAT_GET_X(euler_angles);
    double theta = -GZ_COMPAT_GET_Y(euler_angles);
    double psi = -GZ_COMPAT_GET_Z(euler_angles);
    yaw_desired_ = 0;
    pitch_desired_ = theta;
    roll_desired_ = -phi;
  }

  // Use the Filters to figure out the actual angles
  yaw_actual_ = yaw_filter_->updateFilter(yaw_desired_, dt);
  pitch_actual_ = pitch_filter_->updateFilter(pitch_desired_, dt);
  roll_actual_ = roll_filter_->updateFilter(roll_desired_, dt);

  // Set the Joint Angles to the Filtered angles
#if GAZEBO_MAJOR_VERSION > 5
  yaw_joint_->SetPosition(0, yaw_actual_);
  pitch_joint_->SetPosition(0, pitch_actual_);
  roll_joint_->SetPosition(0, roll_actual_);
#else
  yaw_joint_->SetAngle(0, math::Angle(yaw_actual_));
  pitch_joint_->SetAngle(0, math::Angle(pitch_actual_));
  roll_joint_->SetAngle(0, math::Angle(roll_actual_));
#endif

  // Publish ROS message of actual angles
  geometry_msgs::Vector3Stamped angles_msg;
  angles_msg.header.stamp.sec = GZ_COMPAT_GET_SIM_TIME(world_).sec;
  angles_msg.header.stamp.nsec = GZ_COMPAT_GET_SIM_TIME(world_).nsec;
  angles_msg.vector.x = GZ_COMPAT_GET_POSITION(roll_joint_, 0);
  angles_msg.vector.y = GZ_COMPAT_GET_POSITION(pitch_joint_, 0);
  angles_msg.vector.z = GZ_COMPAT_GET_POSITION(yaw_joint_, 0);
  pose_pub_.publish(angles_msg);
}

void GimbalPlugin::commandCallback(const geometry_msgs::Vector3StampedConstPtr& msg)
{
  // Pull in command from message, convert to NED
  yaw_desired_ = -1.0*msg->vector.z;
  pitch_desired_ = -1.0*msg->vector.y;
  roll_desired_ = msg->vector.x;

  if (!use_slipring_) {
    // Wrap Commands between -PI and PI if a slipring isn't being simulated
    while (fabs(yaw_desired_) > PI) {
      yaw_desired_ -= sign(yaw_desired_)*2.0*PI;
    }
    while (fabs(pitch_desired_) > PI){
      pitch_desired_ -= sign(pitch_desired_)*2.0*PI;
    }
    while (fabs(roll_desired_) > PI) {
      roll_desired_ -= sign(roll_desired_)*2.0*PI;
    }
  }
}

int GimbalPlugin::sign(double x){
  return (0 < x) - (x < 0);
}

GZ_REGISTER_MODEL_PLUGIN(GimbalPlugin);

} // namespace
