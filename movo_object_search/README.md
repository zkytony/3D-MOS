# Package for MOVO Object Search

### Creating this package
```
catkin_create_pkg movo_object_search ar_track_alvar std_msgs rospy roscpp pcl_ros pcl_msgs
catkin_make -DCATKIN_WHITELIST_PACKAGES="movo_object_search"
```

### Start up movo

```
roslaunch movo_bringup movo_system.launch
```

### Run movo RVIZ

```
roslaunch movo_viz view_robot.launch
```

Note that to visualize the point cloud, go to Sensing-> PointCloud2, and make sure
the Topic is `/movo_camera/point_cloud/points


## How-tos

### How to start a Gazebo simulation

1. `roslaunch movo_object_search world_bringup.launch`
   
    This command will start the Gazebo simulation (by default, the gui is `true`),
    and other necessary MOVO components. When the Gazebo simulation starts, wait
    for a few seconds until the MOVO dance completes.
    
    By default the environment is `worlds/tabletop_cube.sdf` which is located
    under the `movo_gazebo` package. If you want to change it, edit the default
    value of the parameter `world` in `environment.launch`.
    
### How to start navigation stack

1. `roslaunch movo_object_search navigation.launch`

    This will launch the necessary components for the navigation stack; It
    is just a proxy for `$(find movo_demos)/launch/nav/map_nav.launch` file.
    
    By default the map for the `tabletop_cube` environment is used; The name
    of the map is called `test_simple`, and the `.pgm` and `.yaml` files are
    located under `$(find movo_demos)/map/`. It's very confusing that the world
    models and the map files are not in the same package. But, it is what it is.
    
**Notes**:

* The amcl poses are published under `/amcl_pose`, not /robot_pose.
* The map is served at `/map` topic, as usual.
* In the topological map package, even though there is a camera pose topic argument,
  the code actually does not make use of the camera pose. Therefore I commented
  it out in the launch file for topological map generation.

    
### How to get volumetric observation?

1. Go to `$(movo_object_search)/scripts/observation` and run `python process_pcl.py`.
   If you want to detect AR tag as well `python process_pcl.py -M`. Note that
   due to ROS-related reasons, `python process_pcl.py -M` will only broadcast
   observations if an AR tag is detected. Therefore, I recommend running these
   two together separately, each with a different node name.
   
### Which RVIZ to use?

1. Use the `nav_cloud.rviz` which allows navigation and also visualizes the observation and point cloud.
