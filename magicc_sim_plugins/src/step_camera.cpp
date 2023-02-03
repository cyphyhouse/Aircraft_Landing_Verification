#include <gazebo/common/Plugin.hh>
#include <ros/ros.h>
#include "magicc_sim_plugins/step_camera.h"
#include "magicc_sim_plugins/gz_compat.h"

#include "gazebo_plugins/gazebo_ros_camera.h"

#include <string>

#include <gazebo/sensors/Sensor.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/sensors/SensorManager.hh>
#include <unistd.h>
#include <math.h>

#include <sensor_msgs/Illuminance.h>


using namespace gazebo;

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(StepCamera)

StepCamera::StepCamera(){
}

StepCamera::~StepCamera(){
    GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(_updateConnection);
}

void StepCamera::Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf)
{
    // Make sure the ROS node for Gazebo has already been initialized
    if (!ros::isInitialized())
    {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
      << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
    }

    CameraPlugin::Load(_parent, _sdf);

    // copying from CameraPlugin into GazeboRosCameraUtils
    this->parentSensor_ = this->parentSensor;
    this->width_ = this->width;
    this->height_ = this->height;
    this->depth_ = this->depth;
    this->format_ = this->format;
    this->camera_ = this->camera;

    GazeboRosCameraUtils::Load(_parent, _sdf);

    float worldRate = GZ_COMPAT_GET_PHYSICS(physics::get_world())->GetMaxStepSize();

    # if GAZEBO_MAJOR_VERSION >= 7
        std::string sensor_name = this->parentSensor_->Name();
        this->_updateRate = 1.0 / this->parentSensor_->UpdateRate();
    # else
        std::string sensor_name = this->parentSensor_->GetName();
        this->_updateRate  = 1.0 / this->parentSensor_->GetUpdateRate();
    # endif

    if(std::ceil(this->_updateRate / worldRate) != this->_updateRate / worldRate){
        gzwarn << "The update rate of sensor " << sensor_name << " does not evenly divide into the "
               << "MaxStepSize of the world. This will result in an actual framerate that is slower than requested. "
               << "Consider decreasing the MaxStepSize, or changing the FrameRate or UpdateRate" << "\n";
    }

    // I think putting this at the end helps improve the consistency of "render-then-update-function", but I don't have
    // any super good evidence for that, only trial and error
    _updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&StepCamera::OnUpdate, this, _1));
    _sensorUpdateConnection = this->parentSensor_->ConnectUpdated(boost::bind(&StepCamera::OnUpdateParentSensor, this));

    // I am not sure why Reset() wasn't being called on it's own. I can only guess that I have it set up
    // wrong somewhere.
    _resetConnection = event::Events::ConnectWorldReset(boost::bind(&StepCamera::Reset, this));
}

void StepCamera::OnUpdateParentSensor(){
    this->_updateLock.unlock(); // We need both of these to get maximial speed.
}

void StepCamera::OnUpdate(const common::UpdateInfo&){
   if(this->parentSensor->IsActive() && (GZ_COMPAT_GET_SIM_TIME(this->world_) - this->last_update_time_) >= (this->_updateRate) ){
     // If we should have published a message, try and get a lock to wait for onUpdateParentSensor
     if(!this->_updateLock.timed_lock(boost::posix_time::seconds(int(this->_updateRate)))){
         ROS_FATAL_STREAM("Update loop timed out waiting for the renderer.");
     }
   }
}

void StepCamera::Reset(){
    this->last_update_time_ = 0;
    this->sensor_update_time_ = 0;
}


void StepCamera::OnNewFrame(const unsigned char *_image,
    unsigned int _width, unsigned int _height, unsigned int _depth,
    const std::string &_format)
{
    common::Time current_time = GZ_COMPAT_GET_SIM_TIME(this->world_);

    if (this->parentSensor->IsActive() && (current_time - this->last_update_time_) >= (this->_updateRate))
    {
        this->_updateLock.unlock(); // We need both of these to get maximial speed.
        this->sensor_update_time_ = current_time;

        this->PutCameraData(_image);
        this->PublishCameraInfo();

        this->last_update_time_ = current_time;
    }

}
