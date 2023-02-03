<div align="center">

# Airport Sim package

![Aarhus logo](/media/logo_airport_sim.png "Aarhus airport sim logo")
</div>

## Table of contents

- [Airport sim package README](#airport-sim-package-readme)
  * [Table of contents](#table-of-contents)
  * [Package description](#package-description)
  * [Usage](#usage)
  * [Dynamic light loading](#dynamic-light-loading)
  * [Documentation](#documentation)

## Package description

This package provides a Gazebo world of the Aarhus airport. Apart from just the map, the simulation is aimed at robots reacting to the airport lights. Tasks such as light maintainance and navigation on the air field can be tested in simulation before moving forward in the real world. Since the airport features >500 different light sources, the simulation can't just spawn every light source. Instead, the package comes with a dynamic light loading controller. This controller is responsible for loading lights into the sim and removing them at runtime. Configurations for the robot's position topic and FoV parameters are also exposed. The package contains a few custom Gazebo models for the airport as well as the light effects.

## Usage

To launch the simulation, enter
```console
user@pc:~$ roslaunch airport_sim airport_sim.launch
```
This will also spawn the car into the simulation, load the dynamic light controller as well as the GPS emulator and the car's virtual Ackermann drive controller. 

<div align="center">

![Airport with all lights enabled](/media/airport_full.png "Airport with all lights enabled")
</div>

## Dynamic light loading
To enable the simulation of all lights without the FPS performance hit, the simulation dynamically loads the lights within the robots FoV. If there are more than the maximum amount of allowed lights within drawing distance of the robot, only the closest lights are spawned to preserve simulation performance. For this method to work, the robot has to publish its ground truth data as ROS messages. The maximum number of lights, drawing distance and ground truth odometry message topic can be defined in the [dynamic light loader config file](/airport_sim/config/dynamic_light_load_config.yaml). The code for the controller as well as a more detailed description of its functionality can be found in the [src directory](/airport_sim/src/) as well as the documentation.

## Documentation

The package's code is documented with rosdoc_lite, which is a light weight version of Doxygen for ROS packages. In order to view the documentation, you can open the [index file](/airport_sim/doc/html/index.html) with any browser of your liking. Example: 
```console
user@pc:~/<path_to_airport_sim>$ firefox doc/html/index.html
```
You may also find it helpful to have a look into the [source code](/airport_sim/src) itself.
