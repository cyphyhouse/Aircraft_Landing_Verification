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
cd src

# Currently under src folder 
git submodule init
git submodule update
```
to update all required submodules.
Then in the ``aircraft_landing`` folder, run command 
```
cd ..

# Currently under aircraft_landing folder
catkin_make 
```
Two folders ``./devel`` and ``./build`` will be generated. 

## Obtain pretrained weights
Download the pretrained weights of the keypoint detector from 
```
https://drive.google.com/file/d/1SvxcQDpmdmogz8YMhao6maXSMECIDaeH/view?usp=sharing
```
The pretrained weight should be placed under folder 
```
src/landing_devel/
```
The keypoint detector is already integrated in the repo. It is based on unet and the code for training the detector can be found at
```
https://github.com/cyphyhouse/Pytorch-UNet.git
```

## Running closed loop simulation
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
Then go to folder ``src/landing_devel/NeuReach2/verse`` by command 
```
cd src/landing_devel/NeuReach2/verse
```
and run command 
```
python3 vision_based_simulation3.py
```
to closed loop simulation with vision pipeline. 
