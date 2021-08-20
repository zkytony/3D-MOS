import rospy

def get_param(param):
    
    if rospy.has_param(param):
        return rospy.get_param(param)
    else:
        return rospy.get_param("~" + param)


def get_if_has_param(param):
    
    if rospy.has_param(param):
        return rospy.get_param(param)
    elif rospy.has_param("~" + param):
        return rospy.get_param("~" + param)
    else:
        return None

