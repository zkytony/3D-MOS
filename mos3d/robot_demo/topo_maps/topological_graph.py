# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mos3d.robot_demo.topo_maps.graph import Node, Graph, Edge
from mos3d.robot_demo.conversion import convert, Frame
import json
from mos3d.util import euclidean_dist
import mos3d.util_viz as util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import os
from collections import deque

class TopoNode(Node):
    def __init__(self, id, world_pose):
        self.id = id
        self.pose = world_pose
        self._coords = self.pose  # for visualization
        self._color = "orange"

    def region_pose(self, region_origin):
        return convert(self.pose, Frame.WORLD, Frame.REGION, region_origin=region_origin)

    def search_space_pose(self, region_origin, search_space_resolution):
        return convert(self.pose, Frame.WORLD, Frame.POMDP_SPACE,
                       region_origin=region_origin,
                       search_space_resolution=search_space_resolution)

    def pomdp_space_pose(self, region_origin, search_space_resolution):
        return self.search_space_pose(region_origin, search_space_resolution)

class TopoMap(Graph):

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            data = json.load(f)

        nodes = {}
        for node_id in data["nodes"]:
            node_data = data["nodes"][node_id]
            x, y = node_data["x"], node_data["y"]
            nodes[int(node_id)]= TopoNode(int(node_id), (x,y,0.0))

        edges = {}
        for i, edge in enumerate(data["edges"]):
            node_id1, node_id2 = edge[0], edge[1]
            n1 = nodes[node_id1]
            n2 = nodes[node_id2]
            edges[i] = Edge(i, n1, n2,
                            data=euclidean_dist(n1.pose, n2.pose))

        return TopoMap(edges)

    def closest_node(self, x, y):
        """Given a point at (x,y) in world frame
        find the node that is closest to this point."""
        return min(self.nodes,
                   key=lambda nid: euclidean_dist(self.nodes[nid].pose[:2], (x,y)))

    def navigable(self, nid1, nid2):
        # DFS find path from nid1 to nid2
        stack = deque()
        stack.append(nid1)
        visited = set()
        while len(stack) > 0:
            nid = stack.pop()
            if nid == nid2:
                return True
            for neighbor_nid in self.neighbors(nid):
                if neighbor_nid not in visited:
                    stack.append(neighbor_nid)
                    visited.add(neighbor_nid)
        return False

    #--- Visualizations ---#
    def visualize(self, ax, canonical_map_yaml_path=None, included_nodes=None,
                  dotsize=5, linewidth=1.0,
                  img=None, consider_placeholders=False, show_nids=False):
        """Visualize the topological map `self`. Nodes are colored by labels, if possible.
        If `consider_placeholders` is True, then all placeholders will be colored grey.
        Note that the graph itself may or may not contain placeholders and `consider_placholders`
        is not used to specify that."""
        # Open the yaml file
        with open(canonical_map_yaml_path) as f:
            map_spec = yaml.safe_load(f)
        if img is None:
            img = mpimg.imread(os.path.join(os.path.dirname(canonical_map_yaml_path), map_spec['image']))
        plt.imshow(img, cmap = plt.get_cmap('gray'), origin="lower")

        h, w = img.shape
        util.zoom_rect((w/2, h/2), img, ax, h_zoom_level=3.0, v_zoom_level=4.0)

        # Plot the nodes
        for nid in self.nodes:
            if included_nodes is not None and nid not in included_nodes:
                continue

            nid_text = str(nid) if show_nids else None

            place = self.nodes[nid]
            pose_x, pose_y = place.pose[:2]  # gmapping coordinates
            plot_x, plot_y = util.plot_dot_map(ax, pose_x, pose_y, map_spec, img,
                                               dotsize=dotsize, color=place.color, zorder=2,
                                               linewidth=linewidth, edgecolor='black', label_text=nid_text)

            # Plot the edges
            for neighbor_id in self._conns[nid]:
                if included_nodes is not None and neighbor_id not in included_nodes:
                    continue

                util.plot_line_map(ax, place.pose, self.nodes[neighbor_id].pose, map_spec, img,
                                   linewidth=1, color='black', zorder=1)

def main():
    topo_map_path = "test_map/example.json"
    gmapping_map_path = "test_map/test_simple.yaml"
    topo_map = TopoMap.load(topo_map_path)
    ax = plt.gca()

    for nid in topo_map.nodes:
        print(topo_map.nodes[nid].pose)
        pomdp_pose = topo_map.nodes[nid].pomdp_space_pose((0.0,0.0), 0.3)
        print(pomdp_pose)
        print(convert(pomdp_pose, Frame.POMDP_SPACE, Frame.WORLD,
                      (0.0, 0.0, 0.0), 0.3))
        print("---")

    topo_map.visualize(ax, canonical_map_yaml_path=gmapping_map_path)
    plt.show()

if __name__ == "__main__":
    main()
