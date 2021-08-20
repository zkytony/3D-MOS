#!/usr/bin/env python

import argparse
import sys
from copy import copy
import rospy
import actionlib
import math
import random
import sys
from action.head_and_torso import HeadJTAS, TorsoJTAS
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

def to_rad(deg):
    return math.pi * deg / 180.0

def to_deg(rad):
    return 180.0 * rad / math.pi

def wait_for_torso_height(torso_topic="/movo/torso_controller/state"):
    if torso_topic=="/movo/torso_controller/state":
        msg = rospy.wait_for_message(torso_topic, JointTrajectoryControllerState, timeout=15)
        assert msg.joint_names[0] == 'linear_joint', "Joint is not linear joint (not torso)."
        position = msg.actual.positions[0]
    else:
        assert torso_topic == "/movo/linear_actuator/joint_states"
        msg = rospy.wait_for_message(torso_topic, JointState, timeout=15)
        assert msg.name[0] == 'linear_joint', "Joint is not linear joint (not torso)."        
        position = msg.position[0]
    return position

def wait_for_head():
    head_topic="/movo/head_controller/state"
    msg = rospy.wait_for_message(head_topic, JointTrajectoryControllerState, timeout=15)
    assert msg.joint_names[0] == 'pan_joint', "Joint is not head joints (need pan or tilt)."
    cur_pan = msg.actual.positions[0]
    cur_tilt = msg.actual.positions[1]
    return cur_pan, cur_tilt

def main():
    if len(sys.argv) != 1 and len(sys.argv) != 3 and len(sys.argv) != 5:
        print("Usage %s [-t tilt] [-T torso]" % sys.argv[0])
        return

    tilt = 0
    torso = 0.05
    if len(sys.argv) > 1:
        if sys.argv[1] == "-t":
            tilt = float(sys.argv[2])
        elif sys.argv[1] == "-T":
            torso = float(sys.argv[2])
        if len(sys.argv) > 3:
            if sys.argv[3] == "-t":
                tilt = float(sys.argv[4])
            elif sys.argv[3] == "-T":
                torso = float(sys.argv[4])
    rospy.init_node('movo_search_object_init')
    print("---head---")
    print(tuple(map(to_deg, wait_for_head())))
    HeadJTAS.move(to_rad(0), to_rad(tilt), v=0.8)
    print(tuple(map(to_deg, wait_for_head())))
    
    print("---torso---")
    rostopics = rospy.get_published_topics()
    torso_topic = "/movo/torso_controller/state"
    if torso_topic not in rostopics:
        torso_topic =  "/movo/linear_actuator/joint_states"
    print(wait_for_torso_height(torso_topic=torso_topic))
    TorsoJTAS.move(torso, torso_topic=torso_topic)
    print(wait_for_torso_height(torso_topic=torso_topic))

if __name__ == "__main__":
    main()
