#include "path_manager_example.h"
#include "ros/ros.h"
#include <cmath>

namespace rosplane
{

path_manager_example::path_manager_example() : path_manager_base()
{
  fil_state_ = fillet_state::STRAIGHT;
  dub_state_ = dubin_state::FIRST;
}

void path_manager_example::manage(const params_s &params, const input_s &input, output_s &output)
{

  if (num_waypoints_ < 2)
  {
    ROS_WARN_THROTTLE(4, "No waypoints received! Loitering about origin at 50m");
    output.flag = false;
    output.Va_d = 12;
    output.c[0] = 0.0f;
    output.c[1] = 0.0f;
    output.c[2] = -50.0f;
    output.rho = params.R_min;
    output.lambda = 1;
  }
  else
  {
    if (waypoints_[idx_a_].chi_valid)
    {
      manage_dubins(params, input, output);
    }
    else
    {
      /** Switch the following for flying directly to waypoints, or filleting corners */
      //manage_line(params, input, output);
      manage_fillet(params, input, output);
    }
  }
}

void path_manager_example::manage_line(const params_s &params, const input_s &input, output_s &output)
{

  Eigen::Vector3f p;
  p << input.pn, input.pe, -input.h;

  int idx_b;
  int idx_c;
  if (idx_a_ == num_waypoints_ - 1)
  {
    idx_b = 0;
    idx_c = 1;
  }
  else if (idx_a_ == num_waypoints_ - 2)
  {
    idx_b = num_waypoints_ - 1;
    idx_c = 0;
  }
  else
  {
    idx_b = idx_a_ + 1;
    idx_c = idx_b + 1;
  }

  Eigen::Vector3f w_im1(waypoints_[idx_a_].w);
  Eigen::Vector3f w_i(waypoints_[idx_b].w);
  Eigen::Vector3f w_ip1(waypoints_[idx_c].w);

  output.flag = true;
  output.Va_d = waypoints_[idx_a_].Va_d;
  output.r[0] = w_im1(0);
  output.r[1] = w_im1(1);
  output.r[2] = w_im1(2);
  Eigen::Vector3f q_im1 = (w_i - w_im1).normalized();
  Eigen::Vector3f q_i = (w_ip1 - w_i).normalized();
  output.q[0] = q_im1(0);
  output.q[1] = q_im1(1);
  output.q[2] = q_im1(2);

  Eigen::Vector3f n_i = (q_im1 + q_i).normalized();
  if ((p - w_i).dot(n_i) > 0.0f)
  {
    if (idx_a_ == num_waypoints_ - 1)
      idx_a_ = 0;
    else
      idx_a_++;
  }

}

void path_manager_example::manage_fillet(const params_s &params, const input_s &input, output_s &output)
{
  if (num_waypoints_ < 3) //since it fillets don't make sense between just two points
  {
    manage_line(params, input, output);
    return;
  }

  Eigen::Vector3f p;
  p << input.pn, input.pe, -input.h;

  int idx_b;
  int idx_c;
  if (idx_a_ == num_waypoints_ - 1)
  {
    idx_b = 0;
    idx_c = 1;
  }
  else if (idx_a_ == num_waypoints_ - 2)
  {
    idx_b = num_waypoints_ - 1;
    idx_c = 0;
  }
  else
  {
    idx_b = idx_a_ + 1;
    idx_c = idx_b + 1;
  }

  Eigen::Vector3f w_im1(waypoints_[idx_a_].w);
  Eigen::Vector3f w_i(waypoints_[idx_b].w);
  Eigen::Vector3f w_ip1(waypoints_[idx_c].w);

  float R_min = params.R_min;

  output.Va_d = waypoints_[idx_a_].Va_d;
  output.r[0] = w_im1(0);
  output.r[1] = w_im1(1);
  output.r[2] = w_im1(2);
  Eigen::Vector3f q_im1 = (w_i - w_im1).normalized();
  Eigen::Vector3f q_i = (w_ip1 - w_i).normalized();
  float beta = acosf(-q_im1.dot(q_i));

  Eigen::Vector3f z;
  switch (fil_state_)
  {
  case fillet_state::STRAIGHT:
    output.flag = true;
    output.q[0] = q_im1(0);
    output.q[1] = q_im1(1);
    output.q[2] = q_im1(2);
    output.c[0] = 1;
    output.c[1] = 1;
    output.c[2] = 1;
    output.rho = 1;
    output.lambda = 1;
    z = w_i - q_im1*(R_min/tanf(beta/2.0));
    if ((p - z).dot(q_im1) > 0)
      fil_state_ = fillet_state::ORBIT;
    break;
  case fillet_state::ORBIT:
    output.flag = false;
    output.q[0] = q_i(0);
    output.q[1] = q_i(1);
    output.q[2] = q_i(2);
    Eigen::Vector3f c = w_i - (q_im1 - q_i).normalized()*(R_min/sinf(beta/2.0));
    output.c[0] = c(0);
    output.c[1] = c(1);
    output.c[2] = c(2);
    output.rho = R_min;
    output.lambda = ((q_im1(0)*q_i(1) - q_im1(1)*q_i(0)) > 0 ? 1 : -1);
    z = w_i + q_i*(R_min/tanf(beta/2.0));
    if ((p - z).dot(q_i) > 0)
    {
      if (idx_a_ == num_waypoints_ - 1)
        idx_a_ = 0;
      else
        idx_a_++;
      fil_state_ = fillet_state::STRAIGHT;
    }
    break;
  }
}

void path_manager_example::manage_dubins(const params_s &params, const input_s &input, output_s &output)
{
  Eigen::Vector3f p;
  p << input.pn, input.pe, -input.h;

  output.Va_d = waypoints_[idx_a_].Va_d;
  output.r[0] = 0;
  output.r[1] = 0;
  output.r[2] = 0;
  output.q[0] = 0;
  output.q[1] = 0;
  output.q[2] = 0;
  output.c[0] = 0;
  output.c[1] = 0;
  output.c[2] = 0;

  switch (dub_state_)
  {
  case dubin_state::FIRST:
    dubinsParameters(waypoints_[0], waypoints_[1], params.R_min);
    output.flag = false;
    output.c[0] = dubinspath_.cs(0);
    output.c[1] = dubinspath_.cs(1);
    output.c[2] = dubinspath_.cs(2);
    output.rho = dubinspath_.R;
    output.lambda = dubinspath_.lams;
    if ((p - dubinspath_.w1).dot(dubinspath_.q1) >= 0) // start in H1
    {
      dub_state_ = dubin_state::BEFORE_H1_WRONG_SIDE;
    }
    else
    {
      dub_state_ = dubin_state::BEFORE_H1;
    }
    break;
  case dubin_state::BEFORE_H1:
    output.flag = false;
    output.c[0] = dubinspath_.cs(0);
    output.c[1] = dubinspath_.cs(1);
    output.c[2] = dubinspath_.cs(2);
    output.rho = dubinspath_.R;
    output.lambda = dubinspath_.lams;
    if ((p - dubinspath_.w1).dot(dubinspath_.q1) >= 0) // entering H1
    {
      dub_state_ = dubin_state::STRAIGHT;
    }
    break;
  case dubin_state::BEFORE_H1_WRONG_SIDE:
    output.flag = false;
    output.c[0] = dubinspath_.cs(0);
    output.c[1] = dubinspath_.cs(1);
    output.c[2] = dubinspath_.cs(2);
    output.rho = dubinspath_.R;
    output.lambda = dubinspath_.lams;
    if ((p - dubinspath_.w1).dot(dubinspath_.q1) < 0) // exit H1
    {
      dub_state_ = dubin_state::BEFORE_H1;
    }
    break;
  case dubin_state::STRAIGHT:
    output.flag = true;
    output.r[0] = dubinspath_.w1(0);
    output.r[1] = dubinspath_.w1(1);
    output.r[2] = dubinspath_.w1(2);
    // output.r[0] = dubinspath_.z1(0);
    // output.r[1] = dubinspath_.z1(1);
    // output.r[2] = dubinspath_.z1(2);
    output.q[0] = dubinspath_.q1(0);
    output.q[1] = dubinspath_.q1(1);
    output.q[2] = dubinspath_.q1(2);
    output.rho = 1;
    output.lambda = 1;
    if ((p - dubinspath_.w2).dot(dubinspath_.q1) >= 0) // entering H2
    {
      if ((p - dubinspath_.w3).dot(dubinspath_.q3) >= 0) // start in H3
      {
        dub_state_ = dubin_state::BEFORE_H3_WRONG_SIDE;
      }
      else
      {
        dub_state_ = dubin_state::BEFORE_H3;
      }
    }
    break;
  case dubin_state::BEFORE_H3:
    output.flag = false;
    output.c[0] = dubinspath_.ce(0);
    output.c[1] = dubinspath_.ce(1);
    output.c[2] = dubinspath_.ce(2);
    output.rho = dubinspath_.R;
    output.lambda = dubinspath_.lame;
    if ((p - dubinspath_.w3).dot(dubinspath_.q3) >= 0) // entering H3
    {
      // increase the waypoint pointer
      int idx_b;
      if (idx_a_ == num_waypoints_ - 1)
      {
        idx_a_ = 0;
        idx_b = 1;
      }
      else if (idx_a_ == num_waypoints_ - 2)
      {
        idx_a_++;
        idx_b = 0;
      }
      else
      {
        idx_a_++;
        idx_b = idx_a_ + 1;
      }

      // plan new Dubin's path to next waypoint configuration
      dubinsParameters(waypoints_[idx_a_], waypoints_[idx_b], params.R_min);

      //start new path
      if ((p - dubinspath_.w1).dot(dubinspath_.q1) >= 0) // start in H1
      {
        dub_state_ = dubin_state::BEFORE_H1_WRONG_SIDE;
      }
      else
      {
        dub_state_ = dubin_state::BEFORE_H1;
      }
    }
    break;
  case dubin_state::BEFORE_H3_WRONG_SIDE:
    output.flag = false;
    output.c[0] = dubinspath_.ce(0);
    output.c[1] = dubinspath_.ce(1);
    output.c[2] = dubinspath_.ce(2);
    output.rho = dubinspath_.R;
    output.lambda = dubinspath_.lame;
    if ((p - dubinspath_.w3).dot(dubinspath_.q3) < 0) // exit H3
    {
      dub_state_ = dubin_state::BEFORE_H1;
    }
    break;
  }
}

Eigen::Matrix3f path_manager_example::rotz(float theta)
{
  Eigen::Matrix3f R;
  R << cosf(theta), -sinf(theta), 0,
  sinf(theta),  cosf(theta), 0,
  0,            0, 1;

  return R;
}

float path_manager_example::mo(float in)
{
  float val;
  if (in > 0)
    val = fmod(in, 2.0*M_PI_F);
  else
  {
    float n = floorf(in/2.0/M_PI_F);
    val = in - n*2.0*M_PI_F;
  }
  return val;
}

void path_manager_example::dubinsParameters(const waypoint_s start_node, const waypoint_s end_node, float R)
{
  float ell = sqrtf((start_node.w[0] - end_node.w[0])*(start_node.w[0] - end_node.w[0]) +
                    (start_node.w[1] - end_node.w[1])*(start_node.w[1] - end_node.w[1]));
  if (ell < 2.0*R)
  {
    ROS_ERROR("The distance between nodes must be larger than 2R.");
  }
  else
  {
    dubinspath_.ps(0) = start_node.w[0];
    dubinspath_.ps(1) = start_node.w[1];
    dubinspath_.ps(2) = start_node.w[2];
    dubinspath_.chis = start_node.chi_d;
    dubinspath_.pe(0) = end_node.w[0];
    dubinspath_.pe(1) = end_node.w[1];
    dubinspath_.pe(2) = end_node.w[2];
    dubinspath_.chie = end_node.chi_d;


    Eigen::Vector3f crs = dubinspath_.ps;
    crs(0) += R*(cosf(M_PI_2_F)*cosf(dubinspath_.chis) - sinf(M_PI_2_F)*sinf(dubinspath_.chis));
    crs(1) += R*(sinf(M_PI_2_F)*cosf(dubinspath_.chis) + cosf(M_PI_2_F)*sinf(dubinspath_.chis));
    Eigen::Vector3f cls = dubinspath_.ps;
    cls(0) += R*(cosf(-M_PI_2_F)*cosf(dubinspath_.chis) - sinf(-M_PI_2_F)*sinf(dubinspath_.chis));
    cls(1) += R*(sinf(-M_PI_2_F)*cosf(dubinspath_.chis) + cosf(-M_PI_2_F)*sinf(dubinspath_.chis));
    Eigen::Vector3f cre = dubinspath_.pe;
    cre(0) += R*(cosf(M_PI_2_F)*cosf(dubinspath_.chie) - sinf(M_PI_2_F)*sinf(dubinspath_.chie));
    cre(1) += R*(sinf(M_PI_2_F)*cosf(dubinspath_.chie) + cosf(M_PI_2_F)*sinf(dubinspath_.chie));
    Eigen::Vector3f cle = dubinspath_.pe;
    cle(0) += R*(cosf(-M_PI_2_F)*cosf(dubinspath_.chie) - sinf(-M_PI_2_F)*sinf(dubinspath_.chie));
    cle(1) += R*(sinf(-M_PI_2_F)*cosf(dubinspath_.chie) + cosf(-M_PI_2_F)*sinf(dubinspath_.chie));

    float theta, theta2;
    // compute L1
    theta = atan2f(cre(1) - crs(1), cre(0) - crs(0));
    float L1 = (crs - cre).norm() + R*mo(2.0*M_PI_F + mo(theta - M_PI_2_F) - mo(dubinspath_.chis - M_PI_2_F))
               + R*mo(2.0*M_PI_F + mo(dubinspath_.chie - M_PI_2_F) - mo(theta - M_PI_2_F));

    // compute L2
    ell = (cle - crs).norm();
    theta = atan2f(cle(1) - crs(1), cle(0) - crs(0));
    float L2;
    if (2.0*R > ell)
      L2 = 9999.0f;
    else
    {
      theta2 = theta - M_PI_2_F + asinf(2.0*R/ell);
      L2 = sqrtf(ell*ell - 4.0*R*R) + R*mo(2.0*M_PI_F + mo(theta2) - mo(dubinspath_.chis - M_PI_2_F))
           + R*mo(2.0*M_PI_F + mo(theta2 + M_PI_F) - mo(dubinspath_.chie + M_PI_2_F));
    }

    // compute L3
    ell = (cre - cls).norm();
    theta = atan2f(cre(1) - cls(1), cre(0) - cls(0));
    float L3;
    if (2.0*R > ell)
      L3 = 9999.0f;
    else
    {
      theta2 = acosf(2.0*R/ell);
      L3 = sqrtf(ell*ell - 4*R*R) + R*mo(2.0*M_PI_F + mo(dubinspath_.chis + M_PI_2_F) - mo(theta + theta2))
           + R*mo(2.0*M_PI_F + mo(dubinspath_.chie - M_PI_2_F) - mo(theta + theta2 - M_PI_F));
    }

    // compute L4
    theta = atan2f(cle(1) - cls(1), cle(0) - cls(0));
    float L4 = (cls - cle).norm() + R*mo(2.0*M_PI_F + mo(dubinspath_.chis + M_PI_2_F) - mo(theta + M_PI_2_F))
               + R*mo(2.0*M_PI_F + mo(theta + M_PI_2_F) - mo(dubinspath_.chie + M_PI_2_F));

    // L is the minimum distance
    int idx = 1;
    dubinspath_.L = L1;
    if (L2 < dubinspath_.L)
    {
      dubinspath_.L = L2;
      idx = 2;
    }
    if (L3 < dubinspath_.L)
    {
      dubinspath_.L = L3;
      idx = 3;
    }
    if (L4 < dubinspath_.L)
    {
      dubinspath_.L = L4;
      idx = 4;
    }

    Eigen::Vector3f e1;
    //        e1.zero();
    e1(0) = 1;
    e1(1) = 0;
    e1(2) = 0;
    switch (idx)
    {
    case 1:
      dubinspath_.cs = crs;
      dubinspath_.lams = 1;
      dubinspath_.ce = cre;
      dubinspath_.lame = 1;
      dubinspath_.q1 = (cre - crs).normalized();
      dubinspath_.w1 = dubinspath_.cs + (rotz(-M_PI_2_F)*dubinspath_.q1)*R;
      dubinspath_.w2 = dubinspath_.ce + (rotz(-M_PI_2_F)*dubinspath_.q1)*R;
      break;
    case 2:
      dubinspath_.cs = crs;
      dubinspath_.lams = 1;
      dubinspath_.ce = cle;
      dubinspath_.lame = -1;
      ell = (cle - crs).norm();
      theta = atan2f(cle(1) - crs(1), cle(0) - crs(0));
      theta2 = theta - M_PI_2_F + asinf(2.0*R/ell);
      dubinspath_.q1 = rotz(theta2 + M_PI_2_F)*e1;
      dubinspath_.w1 = dubinspath_.cs + (rotz(theta2)*e1)*R;
      dubinspath_.w2 = dubinspath_.ce + (rotz(theta2 + M_PI_F)*e1)*R;
      break;
    case 3:
      dubinspath_.cs = cls;
      dubinspath_.lams = -1;
      dubinspath_.ce = cre;
      dubinspath_.lame = 1;
      ell = (cre - cls).norm();
      theta = atan2f(cre(1) - cls(1), cre(0) - cls(0));
      theta2 = acosf(2.0*R/ ell);
      dubinspath_.q1 = rotz(theta + theta2 - M_PI_2_F)*e1;
      dubinspath_.w1 = dubinspath_.cs + (rotz(theta + theta2)*e1)*R;
      dubinspath_.w2 = dubinspath_.ce + (rotz(theta + theta2 - M_PI_F)*e1)*R;
      break;
    case 4:
      dubinspath_.cs = cls;
      dubinspath_.lams = -1;
      dubinspath_.ce = cle;
      dubinspath_.lame = -1;
      dubinspath_.q1 = (cle - cls).normalized();
      dubinspath_.w1 = dubinspath_.cs + (rotz(M_PI_2_F)*dubinspath_.q1)*R;
      dubinspath_.w2 = dubinspath_.ce + (rotz(M_PI_2_F)*dubinspath_.q1)*R;
      break;
    }
    dubinspath_.w3 = dubinspath_.pe;
    dubinspath_.q3 = rotz(dubinspath_.chie)*e1;
    dubinspath_.R = R;
  }
}

}//end namespace
