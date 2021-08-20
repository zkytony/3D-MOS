import rospy
import sensor_msgs.point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import message_filters
from aruco_msgs.msg import MarkerArray as ArMarkerArray


def cb(odom, clock):
    import pdb; pdb.set_trace()

def cb2(msg):
    import pdb; pdb.set_trace()

def main():

    sub1 = message_filters.Subscriber("/ground_truth_odom", Odometry)
    sub2 = message_filters.Subscriber("/aruco_marker_publisher/markers", ArMarkerArray)
    ats = message_filters.ApproximateTimeSynchronizer([sub1, sub2],
                                                      5,1)#queue_size=10, slop=1.0)
    ats.registerCallback(cb)
    
    rospy.init_node("movo_stupid_test",
                    anonymous=True, disable_signals=True)
    rospy.spin()

if __name__ == "__main__":
    main()
