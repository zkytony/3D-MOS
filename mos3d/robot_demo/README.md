### Robot Demo

Demo scenario: A mobile robot (MOVO) searches for objects in a 3D world.

The mobile robot navigates in 2D, while can extend its torso which makes its
motion 3D. It also can pan and tilt its camera.


#### Coordinate Conversion
The 3D world is divided into **search regions**. The regions are marked in the 2D
map as rectangular boxes. A POMDP is instantiated for each search region.
Hence, we have the following coordinate conversion chain:
```
world_coordinate <-> search_region_coordinate -> POMDP_search_space_coordinate
```

#### Topological Graph-based Motion Action Space
A topological graph is created on top of the 2D map for navigation. Hence,
this graph expresses the motion action space. Each node on the topological
graph should have an id. The motion actions are then just specified by node ids.
Each node should ground to a world coordinate, and thus can be converted into
other coordinate frames.

##### Macro motion actions
Leverage the idea of template graph decomposition in (Zheng'aaai18). Decompose
the nodes into groups which results in a more coarse grained topological graph.
This graph can then have its corresponding motion action space.

#### Observation conversino
Again, the robot receives volumetric observation with voxels in the world
coordinate frame. Convert such voxel into POMDP coordinate frame for belief update.


## Appendix

### Topological map

#### How to build `sara_mapping`.

Clone the sara_mapping repository. Then, in the workspace root directory:

1. Go to the root of the workspace. `catkin_make`.

    __Troubleshooting__:
    * Could NOT find OpenCV (missing: contrib) (found version "3.3.1")
