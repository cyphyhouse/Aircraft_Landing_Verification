# Aircraft_Landing_Verification

## Require Packages 
squaternion

## Installation Instruction
Create an empty directory with any name ``aircraft_landing`` in thie example. In the folder run command 
```
git clone https://github.com/yixjia/Aircraft_Landing_Verification.git src
```
to clone the repo. Then go to the cloned directory and run command 
```
git submodule init
git submodule update
```
to update all required submodules.
Then in the ``aircraft_landing`` folder, run command 
```
catkin_make 
```
Two folders ``./devel`` and ``./build`` will be generated. 

## Running experiments
After the installation finished, open two terminals. In the first terminal, run command 
```
source devel/setup.bash
```
and then run command 
```
roslaunch rosplane_sim fixedwing.launch  
```
to start the gazebo simulator. 
Then in the second terminal, run command 
```
source devel/setup.bash
```
Then go to folder ``src/landing_devel`` by command 
```
cd src/landing_devel
```
and run command 
```
python3 update_pos.py
```
to start the dynamic and vision pipeline. 