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

#ifndef GAZEBO_ROS_FAST_CAMERA_HH
#define GAZEBO_ROS_FAST_CAMERA_HH

#include <string>
#include <gazebo/plugins/CameraPlugin.hh>
#include <gazebo_plugins/gazebo_ros_camera_utils.h>

namespace gazebo
{
  class StepCamera : public CameraPlugin, GazeboRosCameraUtils
  {
    event::ConnectionPtr _updateConnection;
    event::ConnectionPtr _sensorUpdateConnection;
    event::ConnectionPtr _resetConnection;
    float _updateRate;

    boost::timed_mutex _updateLock;

    public: StepCamera();
    public: ~StepCamera();
    public: void OnUpdate(const common::UpdateInfo&);
    public: void OnUpdateParentSensor();
    public: void OnRender();

    protected: void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);
    protected: void Reset();
    protected: virtual void OnNewFrame(const unsigned char *_image,
                                        unsigned int _width, unsigned int _height,
                                        unsigned int _depth, const std::string &_format);
  };
}
#endif