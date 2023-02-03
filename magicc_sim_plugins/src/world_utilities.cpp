/*
 * Copyright 2016 Robert Pottorff PCC Lab - BYU - Provo, UT
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

#include "magicc_sim_plugins/world_utilities.h"
#include "gazebo/physics/World.hh"
#include "gazebo/physics/Model.hh"
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo {

WorldUtilities::WorldUtilities() : WorldPlugin() {}

WorldUtilities::~WorldUtilities() {
  if (nh_) {
    nh_->shutdown();
    delete nh_;
  }
}

void WorldUtilities::Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
{
  // Connect ROS
  nh_ = new ros::NodeHandle("~");
  step_command_sub_ = nh_->subscribe("step", 1, &WorldUtilities::stepCommandCallback, this);
  randomize_obstacles_service_sub_ = nh_->advertiseService("randomize_obstacles", &WorldUtilities::randomizeObstaclesCommandCallback, this);

  this->world_ = _parent;
}
bool WorldUtilities::randomizeObstaclesCommandCallback(std_srvs::EmptyRequest& request, std_srvs::EmptyResponse& response)
{
    double max_x = 15;
    double min_x = -15;
    double max_y = 40;
    double min_y = -40;
    for (auto const &m : this->world_->GetModels()){
        if(m->GetName().find("obstacle_") == 0){
            math::Pose randomPose(
              (max_x - min_x) * ( (double)rand() / (double)RAND_MAX ) + min_x,
              (max_y - min_y) * ( (double)rand() / (double)RAND_MAX ) + min_y,
              0, 0, 0, 0);
            
            m->SetInitialRelativePose(randomPose);
        }
    }

    return true;
}

void WorldUtilities::stepCommandCallback(const std_msgs::Int16 &msg)
{
    this->world_->SetPaused(true);

    # if GAZEBO_MAJOR_VERSION >= 3
        this->world_->Step(msg.data);
    # else
        this->world_->StepWorld(msg.data);
    # endif
}

GZ_REGISTER_WORLD_PLUGIN(WorldUtilities)

} // namespace

