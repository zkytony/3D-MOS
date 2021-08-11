import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class WaypointApply(object):
    """Note: adapted from code by Yoon;
    Only applies one move base goal and waits for it to be reached."""
    class Status:
        NOT_RUNNING = "not_running"
        RUNNING = "running"
        SUCCESS = "success"
        FAIL = "fail"
        
    def __init__(self, position, orientation, action_name="navigate"):
        # Get an action client
        srv_name = rospy.get_param("~move_base_service_name")  # 'movo_move_base'
        self.client = actionlib.SimpleActionClient(srv_name, MoveBaseAction)
        self.client.wait_for_server()

        self.status = WaypointApply.Status.NOT_RUNNING
        self.action_name = action_name

        # Define the goal
        # print "type(position[0]):", type(position[0])
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0.0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        self.waypoint_execute()

    def waypoint_execute(self):
        self.status = WaypointApply.Status.RUNNING
        self.client.send_goal(self.goal, self.done_cb, self.active_cb, self.feedback_cb) 
        self.client.wait_for_result()
        result = self.client.get_result()
        if str(moveit_error_dict[result.error_code]) != "SUCCESS":
            self.status = WaypointApply.Status.FAIL
            rospy.logerr("Failed to pick, not trying further")
            return
        rospy.loginfo("Waypoint reached.")
        self.status = WaypointApply.Status.SUCCESS

    def active_cb(self):
        rospy.loginfo("Navigation action "+str(self.action_name+1)+" is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        #To print current pose at each feedback:
        #rospy.loginfo("Feedback for goal "+str(self.action_name)+": "+str(feedback))
        rospy.loginfo("Feedback for goal pose "+str(self.action_name+1)+" received")

    def done_cb(self, status, result):
        # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo("Navigation action "+str(self.action_name)+" received a cancel request after it started executing, completed execution!")
        elif status == 3:
            rospy.loginfo("Navigation action "+str(self.action_name)+" reached")
        elif status == 4:
            rospy.loginfo("Navigation action "+str(self.action_name)+" was aborted by the Action Server")
            rospy.signal_shutdown("Navigation action "+str(self.action_name)+" aborted, shutting down!")
        elif status == 5:
            rospy.loginfo("Navigation action "+str(self.action_name)+" has been rejected by the Action Server")
            rospy.signal_shutdown("Navigation action "+str(self.action_name)+" rejected, shutting down!")
        elif status == 8:
            rospy.loginfo("Navigation action "+str(self.action_name)+" received a cancel request before it started executing, successfully cancelled!")

    def motion_stop(self, duration=1.0):
        self._cfg_cmd.gp_cmd = 'GENERAL_PURPOSE_CMD_NONE'
        self._cfg_cmd.gp_param = 0
        self._cfg_cmd.header.stamp = rospy.get_rostime()
        self._cfg_pub.publish(self._cfg_cmd)

        rospy.logdebug("Stopping velocity command to movo base from BaseVelTest class ...")
        try:
            r = rospy.Rate(10)
            start_time = rospy.get_time()
            while (rospy.get_time() - start_time) < duration:
                self._base_vel_pub.publish(Twist())
                r.sleep()
        except Exception as ex:
            print "Message of base motion failed to be published, error message: ", ex.message
            pass
