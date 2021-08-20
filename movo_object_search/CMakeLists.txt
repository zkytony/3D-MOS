cmake_minimum_required(VERSION 2.8.3)
project(movo_object_search)

find_package(catkin REQUIRED COMPONENTS roslaunch rospy)

catkin_python_setup()

catkin_package()

install(
  PROGRAMS
    scripts/init_robot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
