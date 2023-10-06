import rospy 
import numpy as np 
from gazebo_msgs.srv import SetModelState, SetLightProperties
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from scipy.spatial.transform import Rotation

def set_spotlight_properties(splotlight_angle: float) -> None:
    GZ_SET_LIGHT_PROPERTIES = "/gazebo/set_light_properties"
    rospy.wait_for_service(GZ_SET_LIGHT_PROPERTIES)
    yaw = splotlight_angle
    R = Rotation.from_euler('xyz',[0,1.4, yaw])
    x,y,z,w = R.as_quat()
    try:
        set_light_properties_srv = \
            rospy.ServiceProxy(GZ_SET_LIGHT_PROPERTIES, SetLightProperties)
        resp = set_light_properties_srv(
            light_name='spot_light',
            cast_shadows=True,
            diffuse=ColorRGBA(100,100,100,255),
            specular=ColorRGBA(149, 149, 149, 255),
            attenuation_constant=0.1,
            attenuation_linear=0.0,
            attenuation_quadratic=0.0,
            direction=Vector3(0, 0, -1),
            pose=Pose(position=Point(1000, 500, 300), orientation=Quaternion(x,y,z,w))
        )
        # TODO Check response
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s" % e)

if __name__ == "__main__":
    rospy.init_node('change_spotlight')

    set_spotlight_properties(0.5)