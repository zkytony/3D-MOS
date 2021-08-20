#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point
from ros_util import get_param
import json
import os


def make_pose_msg(posit, orien):
    pose = Pose()
    pose.position.x = posit[0]
    pose.position.y = posit[1]
    pose.position.z = posit[2]
    pose.orientation.x = orien[0]
    pose.orientation.y = orien[1]
    pose.orientation.z = orien[2]
    pose.orientation.w = orien[3]
    return pose            

class PublishSearchRegionMarkers:
    def __init__(self,
                 region_origin,
                 search_region_dimension,
                 search_region_resolution,
                 marker_frame="map",
                 marker_topic="/movo_object_search_in_region/region_markers"):
        self._marker_frame = marker_frame
        self._region_size = search_region_dimension * search_region_resolution
        self._region_origin = region_origin
        
        markers_msg = MarkerArray([PublishSearchRegionMarkers.make_marker_msg(self._region_origin,
                                                                              self._region_size,
                                                                              self._marker_frame,
                                                                              single=True)])
        self._pub = rospy.Publisher(marker_topic,
                                    MarkerArray,
                                    queue_size=10,
                                    latch=True)
        self._pub.publish(markers_msg)

    @classmethod
    def publish_all_regions(cls, regions_file,
                            marker_frame="map",
                            marker_topic="/movo_object_search_in_region/region_markers/all"):
        timestamp = rospy.Time.now()
        markers = []
        with open(regions_file) as f:
            data = json.load(f)
            i = 0
            for region_name in data["regions"]:
                region_data = data["regions"][region_name]
                region_origin = tuple(map(float, region_data["origin"][:2]))
                search_space_dimension = int(region_data["dimension"])
                search_space_resolution = float(region_data["resolution"])
                region_size = search_space_dimension * search_space_resolution
                marker_msg = PublishSearchRegionMarkers.make_marker_msg(region_origin,
                                                                        region_size,
                                                                        marker_frame=marker_frame,
                                                                        timestamp=timestamp,
                                                                        marker_id=i)
                markers.append(marker_msg)
                i += 1
        markers_msg = MarkerArray(markers)
        pub = rospy.Publisher(marker_topic,
                              MarkerArray,
                              queue_size=10,
                              latch=True)
        pub.publish(markers_msg)
        return pub


    @classmethod
    def make_marker_msg(cls, region_origin, region_size,
                        marker_frame="map", timestamp=None,
                        marker_id=1, single=False):
        """Convert voxels to Markers message for visualizatoin"""
        if timestamp is None:
            timestamp = rospy.Time.now()
        region_center = (region_origin[0] + region_size / 2.0,
                         region_origin[1] + region_size / 2.0)
        # rectangle
        h = Header()
        h.stamp = timestamp
        h.frame_id = marker_frame
        marker_msg = Marker()
        marker_msg.header = h
        marker_msg.type = 1  # cube
        marker_msg.ns = "search_region"
        marker_msg.id = marker_id
        marker_msg.action = 0 # add an object
        marker_msg.pose = make_pose_msg((region_center[0], region_center[1], 0.0),
                                        [0,0,0,1])
        marker_msg.scale.x = region_size
        marker_msg.scale.y = region_size
        marker_msg.scale.z = 0.02
        if single:
            marker_msg.color.r = 0.1
            marker_msg.color.g = 0.5
            marker_msg.color.b = 0.5
            marker_msg.color.a = 0.4            
        else:
            marker_msg.color.r = 0.7
            marker_msg.color.g = 0.4
            marker_msg.color.b = 0.3
            marker_msg.color.a = 0.4            
        marker_msg.lifetime = rospy.Duration(0)  # forever
        marker_msg.frame_locked = True
        return marker_msg
    

class PublishTopoMarkers:
    """Debug markers include: topological graph; search region."""
    def __init__(self,
                 map_file,
                 resolution,
                 marker_frame="map",
                 marker_topic="/movo_object_search_in_region/topo_markers",
                 publish=True):
        self._marker_frame = marker_frame
        self._nodes = {}
        self._edges = {}
        self._resolution = resolution
        rospy.loginfo(map_file)
        print(map_file)
        print(os.path.exists(map_file))
        with open(map_file) as f:
            data = json.load(f)

        for node_id in data["nodes"]:
            node_data = data["nodes"][node_id]
            x, y = node_data["x"], node_data["y"]
            self._nodes[int(node_id)]= (x,y,0.0)

        for i, edge in enumerate(data["edges"]):
            node_id1, node_id2 = edge[0], edge[1]
            self._edges[i] = (node_id1, node_id2)

        markers_msg = self.make_markers_msg()
        self._pub = rospy.Publisher(marker_topic,
                                    MarkerArray,
                                    queue_size=10,
                                    latch=True)
        if publish:
            self._pub.publish(markers_msg)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def make_markers_msg(self):
        """Convert voxels to Markers message for visualizatoin"""
        timestamp = rospy.Time.now()
        i = 0
        markers = []
        # nodes
        for nid in self._nodes:
            xyz = self._nodes[nid]
            h = Header()
            h.stamp = timestamp
            h.frame_id = self._marker_frame

            # node cylinder
            marker_msg = Marker()
            marker_msg.header = h
            marker_msg.type = 3  # Cylinder
            marker_msg.ns = "topo_map_node"
            marker_msg.id = i; i+=1
            marker_msg.action = 0 # add an object
            marker_msg.pose = make_pose_msg(xyz, [0,0,0,1])
            marker_msg.scale.x = self._resolution
            marker_msg.scale.y = self._resolution
            marker_msg.scale.z = 0.05
            marker_msg.color.r = 0.8
            marker_msg.color.g = 0.5
            marker_msg.color.b = 0.5
            marker_msg.color.a = 0.7            
            marker_msg.lifetime = rospy.Duration(0)  # forever
            marker_msg.frame_locked = True
            markers.append(marker_msg)

            # node id text
            marker_msg = Marker()
            marker_msg.header = h
            marker_msg.type = 9  # TEXT_VIEW_FACING
            marker_msg.ns = "topo_map_node"
            marker_msg.id = i; i+=1
            marker_msg.action = 0 # add an object
            marker_msg.pose = make_pose_msg(xyz, [0,0,0,1])
            marker_msg.scale.x = self._resolution * 4
            marker_msg.scale.y = self._resolution * 4
            marker_msg.scale.z = 0.05
            marker_msg.color.r = 0.9
            marker_msg.color.g = 0.9
            marker_msg.color.b = 0.9
            marker_msg.color.a = 0.9
            marker_msg.text = str(nid)
            marker_msg.lifetime = rospy.Duration(0)  # forever
            marker_msg.frame_locked = True
            markers.append(marker_msg)            

        # edges
        for eid in self._edges:
            nid1, nid2 = self._edges[eid]
            xyz1, xyz2 = self._nodes[nid1], self._nodes[nid2]

            point1 = Point()
            point1.x = xyz1[0]
            point1.y = xyz1[1]
            point1.z = xyz1[2]
            point2 = Point()
            point2.x = xyz2[0]
            point2.y = xyz2[1]
            point2.z = xyz2[2]                        
            
            h = Header()
            h.stamp = timestamp
            h.frame_id = self._marker_frame

            # refer to: http://library.isr.ist.utl.pt/docs/roswiki/rviz(2f)DisplayTypes(2f)Marker.html#Line_Strip_.28LINE_STRIP.3D4.29
            marker_msg = Marker()
            marker_msg.header = h
            marker_msg.type = 4  # Line strip
            marker_msg.ns = "topo_map_edge"
            marker_msg.id = i; i+=1
            marker_msg.action = 0 # add an object
            marker_msg.pose = make_pose_msg((0,0,0), [0,0,0,1])
            marker_msg.scale.x = 0.01
            marker_msg.points = [point1, point2]

            # black
            marker_msg.color.r = 0.0
            marker_msg.color.g = 0.0
            marker_msg.color.b = 0.0
            marker_msg.color.a = 1.0            
            marker_msg.lifetime = rospy.Duration(0)  # forever
            marker_msg.frame_locked = True
            markers.append(marker_msg)            

        marker_array_msg = MarkerArray(markers)
        return marker_array_msg


def main():
    rospy.init_node("topo_map_marker_publisher")

    # # Test topo map markers
    topo_map_file = get_param("topo_map_file")
    resolution = get_param("resolution")
    PublishTopoMarkers(topo_map_file, resolution)
    # rospy.spin()

    regions_file = get_param("regions_file")    
    pub = PublishSearchRegionMarkers.publish_all_regions(regions_file)

    region_name = get_param("region_name")  # specify by _region_name:="shelf-corner"    
    with open(regions_file) as f:
        data = json.load(f)
        region_data = data["regions"][region_name]
    region_origin = tuple(map(float, region_data["origin"][:2]))
    search_space_dimension = int(region_data["dimension"])
    search_space_resolution = float(region_data["resolution"])
    PublishSearchRegionMarkers(region_origin, search_space_dimension,
                               search_space_resolution)
    rospy.spin()

if __name__ == "__main__":
    main()
