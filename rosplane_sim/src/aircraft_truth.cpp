/*
 * Copyright 2016 Gary Ellingson, MAGICC Lab, Brigham Young University, Provo, UT
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

#include "rosplane_sim/aircraft_truth.h"

namespace gazebo
{

AircraftTruth::AircraftTruth() :
  ModelPlugin(),
  nh_(nullptr),
  prev_sim_time_(0)
{}


AircraftTruth::~AircraftTruth()
{
  GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(updateConnection_);
  if (nh_)
  {
    nh_->shutdown();
    delete nh_;
  }
}


void AircraftTruth::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  model_ = _model;
  world_ = model_->GetWorld();

  namespace_.clear();

  /*
   * Connect the Plugin to the Robot and Save pointers to the various elements in the simulation
   */
  if (_sdf->HasElement("namespace"))
    namespace_ = _sdf->GetElement("namespace")->Get<std::string>();
  else
    gzerr << "[gazebo_aircraft_truth] Please specify a namespace.\n";
  nh_ = new ros::NodeHandle(namespace_);

  if (_sdf->HasElement("linkName"))
    link_name_ = _sdf->GetElement("linkName")->Get<std::string>();
  else
    gzerr << "[gazebo_aircraft_truth] Please specify a linkName of the truth plugin.\n";
  link_ = model_->GetLink(link_name_);
  if (link_ == NULL)
    gzthrow("[gazebo_aircraft_truth] Couldn't find specified link \"" << link_name_ << "\".");

  /* Load Params from Gazebo Server */
  wind_speed_topic_ = nh_->param<std::string>("windSpeedTopic", "gazebo/wind_speed");
  truth_topic_ = nh_->param<std::string>("truthTopic", "truth");

  // Connect the update function to the simulation
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&AircraftTruth::OnUpdate, this, _1));

  // Connect Subscribers
  true_state_pub_ = nh_->advertise<rosplane_msgs::State>(truth_topic_, 1);
  wind_speed_sub_ = nh_->subscribe(wind_speed_topic_, 1, &AircraftTruth::WindSpeedCallback, this);
}

// This gets called by the world update event.
void AircraftTruth::OnUpdate(const common::UpdateInfo &_info)
{

  sampling_time_ = _info.simTime.Double() - prev_sim_time_;
  prev_sim_time_ = _info.simTime.Double();
  PublishTruth();
}

void AircraftTruth::WindSpeedCallback(const geometry_msgs::Vector3 &wind)
{
  wind_.N = wind.x;
  wind_.E = wind.y;
  wind_.D = wind.z;
}


void AircraftTruth::PublishTruth()
{
  /* Get state information from Gazebo - convert to NED         *
   * C denotes child frame, P parent frame, and W world frame.  *
   * Further C_pose_W_P denotes pose of P wrt. W expressed in C.*/
  rosplane_msgs::State msg;
  // Set origin values to zero by default
  msg.initial_lat = 0;
  msg.initial_lon = 0;
  msg.initial_alt = 0;

  GazeboPose W_pose_W_C = GZ_COMPAT_GET_WORLD_COG_POSE(link_);
  msg.position[0] = GZ_COMPAT_GET_X(GZ_COMPAT_GET_POS(W_pose_W_C)); // We should check to make sure that this is right
  msg.position[1] = -GZ_COMPAT_GET_Y(GZ_COMPAT_GET_POS(W_pose_W_C));
  msg.position[2] = -GZ_COMPAT_GET_Z(GZ_COMPAT_GET_POS(W_pose_W_C));
  GazeboVector euler_angles = GZ_COMPAT_GET_EULER(GZ_COMPAT_GET_ROT(W_pose_W_C));
  msg.phi = GZ_COMPAT_GET_X(euler_angles);
  msg.theta = -GZ_COMPAT_GET_Y(euler_angles);
  msg.psi = -GZ_COMPAT_GET_Z(euler_angles);
  GazeboVector C_linear_velocity_W_C = GZ_COMPAT_GET_RELATIVE_LINEAR_VEL(link_);
  double u = GZ_COMPAT_GET_X(C_linear_velocity_W_C);
  double v = -GZ_COMPAT_GET_Y(C_linear_velocity_W_C);
  double w = -GZ_COMPAT_GET_Z(C_linear_velocity_W_C);
  msg.Vg = sqrt(pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0));
  GazeboVector C_angular_velocity_W_C = GZ_COMPAT_GET_RELATIVE_ANGULAR_VEL(link_);
  msg.p = GZ_COMPAT_GET_X(C_angular_velocity_W_C);
  msg.q = -GZ_COMPAT_GET_Y(C_angular_velocity_W_C);
  msg.r = -GZ_COMPAT_GET_Z(C_angular_velocity_W_C);

  msg.wn = wind_.N;
  msg.we = wind_.E;

  // wind info is available in the wind_ struct
  double ur = u ;//- wind_.N;
  double vr = v ;//- wind_.E;
  double wr = w ;//- wind_.D;

  msg.Va = sqrt(pow(ur, 2.0) + pow(vr, 2.0) + pow(wr, 2.0));
  msg.chi = atan2(msg.Va*sin(msg.psi), msg.Va*cos(msg.psi));
  msg.alpha = atan2(wr , ur);
  msg.beta = asin(vr/msg.Va);

  msg.quat_valid = false;
  msg.quat[0] = u;
  msg.quat[1] = v;
  msg.quat[2] = w;

  msg.header.stamp.fromSec(GZ_COMPAT_GET_SIM_TIME(world_).Double());
  msg.header.frame_id = 1; // Denotes global frame

  msg.psi_deg = fmod(GZ_COMPAT_GET_X(euler_angles), 2.0*M_PI)*180.0 / M_PI; //-360 to 360
  msg.psi_deg += (msg.psi_deg < -180.0 ? 360.0 : 0.0);
  msg.psi_deg -= (msg.psi_deg > 180.0 ? 360.0 : 0.0);
  msg.chi_deg = fmod(msg.chi, 2.0*M_PI)*180.0 / M_PI; //-360 to 360
  msg.chi_deg += (msg.chi_deg < -180.0 ? 360.0 : 0.0);
  msg.chi_deg -= (msg.chi_deg > 180.0 ? 360.0 : 0.0);

  true_state_pub_.publish(msg);
}

GZ_REGISTER_MODEL_PLUGIN(AircraftTruth);
}
