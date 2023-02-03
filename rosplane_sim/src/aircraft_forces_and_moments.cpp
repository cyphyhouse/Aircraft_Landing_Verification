/*
 * Copyright 2016 James Jackson, MAGICC Lab, Brigham Young University, Provo, UT
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

#include "rosplane_sim/aircraft_forces_and_moments.h"

namespace gazebo
{

AircraftForcesAndMoments::AircraftForcesAndMoments() {}


AircraftForcesAndMoments::~AircraftForcesAndMoments()
{
  GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(updateConnection_);
  if (nh_)
  {
    nh_->shutdown();
    delete nh_;
  }
}

void AircraftForcesAndMoments::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
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
    gzerr << "[gazebo_aircraft_forces_and_moments] Please specify a namespace.\n";
  nh_ = new ros::NodeHandle(namespace_);

  if (_sdf->HasElement("linkName"))
    link_name_ = _sdf->GetElement("linkName")->Get<std::string>();
  else
    gzerr << "[gazebo_aircraft_forces_and_moments] Please specify a linkName of the forces and moments plugin.\n";
  link_ = model_->GetLink(link_name_);
  if (link_ == NULL)
    gzthrow("[gazebo_aircraft_forces_and_moments] Couldn't find specified link \"" << link_name_ << "\".");

  /* Load Params from Gazebo Server */
  wind_speed_topic_ = nh_->param<std::string>("windSpeedTopic", "wind");
  command_topic_ = nh_->param<std::string>("commandTopic", "command");

  // The following parameters are aircraft-specific, most of these can be found using AVL
  // The rest are more geometry-based and can be found in conventional methods
  // For the moments of inertia, look into using the BiFilar pendulum method

  // physical parameters
  mass_ = nh_->param<double>("mass", 3.92);
  Jx_ = nh_->param<double>("Jx", 0.213);
  Jy_ = nh_->param<double>("Jy", 0.171);
  Jz_ = nh_->param<double>("Jz", 0.350);
  Jxz_ = nh_->param<double>("Jxz", 0.04);
  rho_ = nh_->param<double>("rho", 1.268);

  // Wing Geometry
  wing_.S = nh_->param<double>("wing_s", 0.468);
  wing_.b = nh_->param<double>("wing_b", 1.8);
  wing_.c = nh_->param<double>("wing_c", 0.26);
  wing_.M = nh_->param<double>("wing_M", 50);
  wing_.epsilon = nh_->param<double>("wing_epsilon", 0.159);
  wing_.alpha0 = nh_->param<double>("wing_alpha0", 0.304);

  // Propeller Coefficients
  prop_.k_motor = nh_->param<double>("k_motor", 40.0);
  prop_.k_T_P = nh_->param<double>("k_T_P", 0.0);
  prop_.k_Omega = nh_->param<double>("k_Omega", 0.0);
  prop_.e = nh_->param<double>("prop_e", 0.8);
  prop_.S = nh_->param<double>("prop_S", 0.0314);
  prop_.C = nh_->param<double>("prop_C", 1.0);

  // Lift Params
  CL_.O = nh_->param<double>("C_L_O", 0.2869);
  CL_.alpha = nh_->param<double>("C_L_alpha", 5.1378);
  CL_.beta = nh_->param<double>("C_L_beta", 0.0);
  CL_.p = nh_->param<double>("C_L_p", 0.0);
  CL_.q = nh_->param<double>("C_L_q", 1.7102);
  CL_.r = nh_->param<double>("C_L_r", 0.0);
  CL_.delta_a = nh_->param<double>("C_L_delta_a", 0.0);
  CL_.delta_e = nh_->param<double>("C_L_delta_e", 0.5202);
  CL_.delta_r = nh_->param<double>("C_L_delta_r", 0.0);

  // Drag Params
  CD_.O = nh_->param<double>("C_D_O", 0.03087);
  CD_.alpha = nh_->param<double>("C_D_alpha", 0.0043021);
  CD_.beta = nh_->param<double>("C_D_beta", 0.0);
  CD_.p = nh_->param<double>("C_D_p", 0.02815);
  CD_.q = nh_->param<double>("C_D_q", 0.2514);
  CD_.r = nh_->param<double>("C_D_r", 0.0);
  CD_.delta_a = nh_->param<double>("C_D_delta_a", 0.0);
  CD_.delta_e = nh_->param<double>("C_D_delta_e", 0.01879);
  CD_.delta_r = nh_->param<double>("C_D_delta_r", 0.0);

  // ell Params (x axis moment)
  Cell_.O = nh_->param<double>("C_ell_O", 0.0);
  Cell_.alpha = nh_->param<double>("C_ell_alpha", 0.00);
  Cell_.beta = nh_->param<double>("C_ell_beta", 0.0193);
  Cell_.p = nh_->param<double>("C_ell_p", -0.5406);
  Cell_.q = nh_->param<double>("C_ell_q", 0.0);
  Cell_.r = nh_->param<double>("C_ell_r", 0.1929);
  Cell_.delta_a = nh_->param<double>("C_ell_delta_a", 0.2818);
  Cell_.delta_e = nh_->param<double>("C_ell_delta_e", 0.0);
  Cell_.delta_r = nh_->param<double>("C_ell_delta_r", 0.00096);

  // m Params (y axis moment)
  Cm_.O = nh_->param<double>("C_m_O", 0.0362);
  Cm_.alpha = nh_->param<double>("C_m_alpha", -0.2627);
  Cm_.beta = nh_->param<double>("C_m_beta", 0.0);
  Cm_.p = nh_->param<double>("C_m_p", 0.0);
  Cm_.q = nh_->param<double>("C_m_q", -9.7213);
  Cm_.r = nh_->param<double>("C_m_r", 0.0);
  Cm_.delta_a = nh_->param<double>("C_m_delta_a", 0.0);
  Cm_.delta_e = nh_->param<double>("C_m_delta_e", -1.2392);
  Cm_.delta_r = nh_->param<double>("C_m_delta_r", 0.0);

  // n Params (z axis moment)
  Cn_.O = nh_->param<double>("C_n_O", 0.0);
  Cn_.alpha = nh_->param<double>("C_n_alpha", 0.0);
  Cn_.beta = nh_->param<double>("C_n_beta", 0.08557);
  Cn_.p = nh_->param<double>("C_n_p", -0.0498);
  Cn_.q = nh_->param<double>("C_n_q", 0.0);
  Cn_.r = nh_->param<double>("C_n_r", -0.0572);
  Cn_.delta_a = nh_->param<double>("C_n_delta_a", 0.0095);
  Cn_.delta_e = nh_->param<double>("C_n_delta_e", 0.0);
  Cn_.delta_r = nh_->param<double>("C_n_delta_r", -0.06);

  // Y Params (Sideslip Forces)
  CY_.O = nh_->param<double>("C_Y_O", 0.0);
  CY_.alpha = nh_->param<double>("C_Y_alpha", 0.00);
  CY_.beta = nh_->param<double>("C_Y_beta", -0.2471);
  CY_.p = nh_->param<double>("C_Y_p", -0.07278);
  CY_.q = nh_->param<double>("C_Y_q", 0.0);
  CY_.r = nh_->param<double>("C_Y_r", 0.1849);
  CY_.delta_a = nh_->param<double>("C_Y_delta_a", -0.02344);
  CY_.delta_e = nh_->param<double>("C_Y_delta_e", 0.0);
  CY_.delta_r = nh_->param<double>("C_Y_delta_r", 0.1591);

  // Initialize Wind
  wind_.N = 0.0;
  wind_.E = 0.0;
  wind_.D = 0.0;

  //initialize deltas
  delta_.t = 0.0;
  delta_.e = 0.0;
  delta_.a = 0.0;
  delta_.r = 0.0;

  // Connect the update function to the simulation
  updateConnection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&AircraftForcesAndMoments::OnUpdate, this, _1));

  // Connect Subscribers
  command_sub_ = nh_->subscribe(command_topic_, 1, &AircraftForcesAndMoments::CommandCallback, this);
  wind_speed_sub_ = nh_->subscribe(wind_speed_topic_, 1, &AircraftForcesAndMoments::WindSpeedCallback, this);

  // Pull off initial pose so we can reset to it
  initial_pose_ = GZ_COMPAT_GET_WORLD_COG_POSE(link_);
}

// This gets called by the world update event.
void AircraftForcesAndMoments::OnUpdate(const common::UpdateInfo &_info)
{
  sampling_time_ = _info.simTime.Double() - prev_sim_time_;
  prev_sim_time_ = _info.simTime.Double();
  UpdateForcesAndMoments();
  SendForces();
}

void AircraftForcesAndMoments::Reset()
{
  forces_.Fx = 0.0;
  forces_.Fy = 0.0;
  forces_.Fz = 0.0;
  forces_.l = 0.0;
  forces_.m = 0.0;
  forces_.n = 0.0;

  link_->SetWorldPose(initial_pose_);
  link_->ResetPhysicsStates();
}

void AircraftForcesAndMoments::WindSpeedCallback(const geometry_msgs::Vector3 &wind)
{
  wind_.N = wind.x;
  wind_.E = wind.y;
  wind_.D = wind.z;
}

void AircraftForcesAndMoments::CommandCallback(const rosflight_msgs::CommandConstPtr &msg)
{
  // This is a little bit weird.  We need to nail down why these are negative
  delta_.t = msg->F;
  delta_.e = -msg->y;
  delta_.a = msg->x;
  delta_.r = -msg->z;
}


void AircraftForcesAndMoments::UpdateForcesAndMoments()
{
  /* Get state information from Gazebo (in NED)                 *
   * C denotes child frame, P parent frame, and W world frame.  *
  //   * Further C_pose_W_P denotes pose of P wrt. W expressed in C.*/
  GazeboVector C_linear_velocity_W_C = GZ_COMPAT_GET_RELATIVE_LINEAR_VEL(link_);
  double u = GZ_COMPAT_GET_X(C_linear_velocity_W_C);
  double v = -GZ_COMPAT_GET_Y(C_linear_velocity_W_C);
  double w = -GZ_COMPAT_GET_Z(C_linear_velocity_W_C);
  GazeboVector C_angular_velocity_W_C = GZ_COMPAT_GET_RELATIVE_ANGULAR_VEL(link_);
  double p = GZ_COMPAT_GET_X(C_angular_velocity_W_C);
  double q = -GZ_COMPAT_GET_Y(C_angular_velocity_W_C);
  double r = -GZ_COMPAT_GET_Z(C_angular_velocity_W_C);

  // wind info is available in the wind_ struct
  /// TODO: This is wrong. Wind is being applied in the body frame, not inertial frame
  double ur = u - wind_.N;
  double vr = v - wind_.E;
  double wr = w - wind_.D;

  double Va = sqrt(pow(ur, 2.0) + pow(vr, 2.0) + pow(wr, 2.0));

  // Don't divide by zero, and don't let NaN's get through (sometimes GetRelativeLinearVel returns NaNs)
  if (Va > 0.000001 && std::isfinite(Va))
  {
    /*
     * The following math follows the method described in chapter 4 of
     * Small Unmanned Aircraft: Theory and Practice
     * By Randy Beard and Tim McLain.
     * Look there for a detailed explanation of each line in the rest of this function
     */
    double alpha = atan2(wr , ur);
    double beta = asin(vr/Va);

    double sign = (alpha >= 0 ? 1 : -1); //Sigmoid function
    double sigma_a = (1 + exp(-(wing_.M*(alpha - wing_.alpha0))) + exp((wing_.M*(alpha + wing_.alpha0))))/((1 + exp(-
                     (wing_.M*(alpha - wing_.alpha0))))*(1 + exp((wing_.M*(alpha + wing_.alpha0)))));
    double CL_a = (1 - sigma_a)*(CL_.O + CL_.alpha*alpha) + sigma_a*(2.0*sign*pow(sin(alpha), 2.0)*cos(alpha));
    double AR = (pow(wing_.b, 2.0))/wing_.S;
    double CD_a = CD_.p + ((pow((CL_.O + CL_.alpha*(alpha)),
                                2.0))/(3.14159*0.9 *
                                         AR)); //the const 0.9 in this equation replaces the e (Oswald Factor) variable and may be inaccurate

    double CX_a = -CD_a*cos(alpha) + CL_a*sin(alpha);
    double CX_q_a = -CD_.q*cos(alpha) + CL_.q*sin(alpha);
    double CX_deltaE_a = -CD_.delta_e*cos(alpha) + CL_.delta_e*sin(alpha);

    double CZ_a = -CD_a*sin(alpha) - CL_a*cos(alpha);
    double CZ_q_a = -CD_.q*sin(alpha) - CL_.q*cos(alpha);
    double CZ_deltaE_a = -CD_.delta_e*sin(alpha) - CL_.delta_e*cos(alpha);

    forces_.Fx = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*(CX_a + (CX_q_a*wing_.c*q) /
                 (2.0*Va) + CX_deltaE_a*delta_.e) + 0.5*rho_*prop_.S*prop_.C*(pow((prop_.k_motor*delta_.t), 2.0) - pow(Va,
                     2.0));
    forces_.Fy = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*(CY_.O + CY_.beta*beta + ((CY_.p*wing_.b*p) /
                 (2.0*Va)) + ((CY_.r*wing_.b*r)/(2.0*Va)) + CY_.delta_a*delta_.a + CY_.delta_r*delta_.r);
    forces_.Fz = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*(CZ_a + (CZ_q_a*wing_.c*q) /
                 (2.0*Va) + CZ_deltaE_a*delta_.e);

    forces_.l = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*wing_.b*(Cell_.O + Cell_.beta*beta + (Cell_.p*wing_.b*p) /
                (2.0*Va) + (Cell_.r*wing_.b*r)/(2.0*Va) + Cell_.delta_a*delta_.a + Cell_.delta_r*delta_.r) - prop_.k_T_P *
                pow((prop_.k_Omega*delta_.t), 2.0);
    forces_.m = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*wing_.c*(Cm_.O + Cm_.alpha*alpha + (Cm_.q*wing_.c*q) /
                (2.0*Va) + Cm_.delta_e*delta_.e);
    forces_.n = 0.5*(rho_)*pow(Va, 2.0)*wing_.S*wing_.b*(Cn_.O + Cn_.beta*beta + (Cn_.p*wing_.b*p) /
                (2.0*Va) + (Cn_.r*wing_.b*r)/(2.0*Va) + Cn_.delta_a*delta_.a + Cn_.delta_r*delta_.r);
  }
  else
  {
    if (!std::isfinite(Va))
    {
      gzerr << "u = " << u << "\n";
      gzerr << "v = " << v << "\n";
      gzerr << "w = " << w << "\n";
      gzerr << "p = " << p << "\n";
      gzerr << "q = " << q << "\n";
      gzerr << "r = " << r << "\n";
      gzerr << "ur = " << ur << "\n";
      gzerr << "vr = " << vr << "\n";
      gzerr << "wr = " << wr << "\n";
      gzthrow("we have a NaN or an infinity:\n");
    }
    else
    {
      forces_.Fx = 0.5*rho_*prop_.S*prop_.C*(pow((prop_.k_motor*delta_.t), 2.0));
      forces_.Fy = 0.0;
      forces_.Fz = 0.0;
      forces_.l = 0.0;
      forces_.m = 0.0;
      forces_.n = 0.0;
    }
  }
}


void AircraftForcesAndMoments::SendForces()
{
  // Make sure we are applying reasonable forces
  if (std::isfinite(forces_.Fx + forces_.Fy + forces_.Fz + forces_.l + forces_.m + forces_.n))
  {
    // apply the forces and torques to the joint
    link_->AddRelativeForce(GazeboVector(forces_.Fx, -forces_.Fy, -forces_.Fz));
    link_->AddRelativeTorque(GazeboVector(forces_.l, -forces_.m, -forces_.n));
  }
}

GZ_REGISTER_MODEL_PLUGIN(AircraftForcesAndMoments);
}
