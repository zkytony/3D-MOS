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

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import *
from mos3d.models.world.objects import GridWorldObject
import mos3d.util as util
import numpy as np

class Robot(GridWorldObject):
    # By default, the robot has a camera, and
    # it looks into the direction (-1, 0, 0)
    def __init__(self, id_,
                 camera_pose,  # 6D camera pose relative to the robot
                 camera_model,
                 objtype="robot"):

        super().__init__(id_, objtype=objtype)
        self._camera_pose = camera_pose
        self._camera_model = camera_model
        sx, sy, sz, sthx, sthy, sthz = self._camera_pose
        self._camera_model.transform_camera(self._camera_pose, permanent=True) #)sx, sy, sz, sthx, sthy, sthz, permanent=True)

    def init_render(self):
        vertices, indices, colors = util.cube(color=(1,0,0))
        self._vertex_vbo, self._index_vbo, self._color_vbo\
            = util.generate_vbo_elements(vertices, indices, colors)
        self._num_indices = len(indices)

        # vertices for axes
        axes_vertices = np.array([0,0,0,
                                  0,0,0,
                                  0,0,0,
                                  2,0,0,
                                  0,2,0,
                                  0,0,2])
        axes_colors = np.array([0.8,0.2,0.2,  # origin - red
                                0.2,0.8,0.2,  # origin - green
                                0.2,0.2,0.8,  # origin - blue
                                0.8,0.2,0.2,  # Red
                                0.2,0.8,0.2,  # Green
                                0.2,0.2,0.8]) # Blue
        axes_indices = np.array([0,3,1,4,2,5])
        self._axes_vertex_vbo, self._axes_index_vbo, self._axes_color_vbo \
            = util.generate_vbo_elements(axes_vertices, axes_indices, axes_colors)

        self._camera_model.init_render()

    def _render_axes(self):
        # Draw axes
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, self._axes_vertex_vbo);
        glVertexPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ARRAY_BUFFER, self._axes_color_vbo);
        glColorPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._axes_index_vbo);
        glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_COLOR_ARRAY)

    def render(self, render_fov=False):
        if render_fov:
            glPushMatrix()
            sx, sy, sz, sthx, sthy, sthz = self._camera_pose
            glTranslatef(sx, sy, sz)
            glRotatef(sthz, 0, 0, 1)
            glRotatef(sthy, 0, 1, 0)
            glRotatef(sthx, 1, 0, 0)
            self._camera_model.render()
            glPopMatrix()

        # render robot cube
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, self._vertex_vbo);
        glVertexPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo);
        glColorPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_vbo);
        glDrawElements(GL_QUADS, self._num_indices, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_COLOR_ARRAY)

        # render axis
        self._render_axes()


    def cleanup(self):
        glDeleteBuffers(3, np.array([self._vertex_vbo, self._index_vbo, self._color_vbo]))

    @property
    def camera_model(self):
        return self._camera_model

    def camera_pose(self, robot_pose):
        """
        Returns world-frame camera pose, with rotation in quaternion.
        """
        # robot pose with respect to world frame
        rx, ry, rz, qx, qy, qz, qw = robot_pose
        rthx, rthy, rthz = util.quat_to_euler(qx, qy, qz, qw)
        # camera pose with respect to robot frame
        sx, sy, sz, sthx, sthy, sthz = self._camera_pose
        # camera pose with respect to world frame
        cam_pose_world = [rx + sx, ry + sy, rz + sz]\
            + util.R_to_quat(util.R_quat(qx, qy, qz, qw) * util.R_euler(sthx, sthy, sthz)).tolist()
        return cam_pose_world
