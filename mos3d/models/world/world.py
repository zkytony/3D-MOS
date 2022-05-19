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
from OpenGL.arrays import vbo
import random
import numpy as np

from mos3d.models.world.objects import *
from mos3d.models.world.robot import Robot
from mos3d.models.world.sensor_model import FrustumCamera
from mos3d.models.observation import OOObservation, Voxel
import mos3d.util as util

OBJECT_MANAGER = GObjManager()
OBJECT_MANAGER.register_all([(Robot, 1, 'robot'),
                             (Cube, 2, 'cube'),
                             (OrangeRicky, 3, 'orange_ricky'),
                             (Hero, 4, 'hero'),
                             (Teewee, 5, 'teewee'),
                             (Smashboy, 6, 'smashboy')])

def diff(rang):
    return rang[1] - rang[0]

class GridWorld:

    """Gridworld provides implementation of motion and observation functions
    but does not store the true state. The true state is only stored by
    the Environment. The Gridworld does require general information such
    as objects (their ids, types) and world boundary."""

    def __init__(self, w, l, h, robot, objects,
                 robot_id=0, occlusion_enabled=False,
                 obstacles=set({}), hidden=set({})):
        """
        robot (Robot); no pose or other variable information is stored
        objects (dict) {id -> obj(Object); no pose is stored }.
        obstacles (set): set of object ids (subset of objects.keys()) that
                         represent objects which are obstacles.
        hidden (set): set of grid locations (x,y,z) that will always be UNKNOWN,
                      as if it is occluded.
        """
        self._w, self._l, self._h = w, l, h
        self._robot_id = robot_id
        self._robot = robot
        self._objects = objects
        self._obstacles = obstacles
        self._hidden = hidden
        self._target_objects = set(objects.keys()) - obstacles
        self._x_range = (0, self._w-1)
        self._y_range = (0, self._l-1)
        self._z_range = (0, self._h-1)
        self._occlusion_enabled = occlusion_enabled
        self._observation_cache = {}  # maps from (robot_pose, object_poses) to {set of observable objects}

    def valid_pose(self, pose, object_poses=None, check_collision=True):
        x, y, z = pose[:3]

        # Check collision
        if check_collision and object_poses is not None:
            for objid in object_poses:
                true_object_pose = object_poses[objid]
                for cube_pose in self._objects[objid].cube_poses(*true_object_pose):
                    if (x,y,z) == tuple(cube_pose):
                        return False
        return self.in_boundary(pose)

    def is_obstacle(self, objid):
        return objid in self._obstacles

    @property
    def target_objects(self):
        # Returns a set of objids for target objects
        return self._target_objects

    def in_boundary(self, pose):
        # Check if in boundary
        x,y,z = pose[:3]
        if x >= 0 and x < self.width:
            if y >= 0 and y < self.length:
                if z >= 0 and z < self.height:
                    if len(pose) > 3 and len(pose) < 7:
                        # check if orientation is valid
                        thx, thy, thz = pose[3:]
                        if thx >= 0 and thx <= 360:
                            if thy >= 0 and thy <= 360:
                                if thz >= 0 and thz <= 360:
                                    return True
                    elif len(pose) == 7:
                        # check if quaternion is valid (unorm=1)
                        qx, qy, qz, qw = pose[3:]
                        return abs((np.linalg.norm([qx,qy,qz,qw])) - 1.0) <= 1e-6
                    else:
                        return True
        return False

    # MOTION MODEL
    def if_move_by(self, *params, motion_model="AXIS", valid_pose_func=None, object_poses=None,
                   absolute_rotation=False):
        if motion_model == "AXIS":
            return self.if_move_by_axis(*params, valid_pose_func, object_poses=object_poses, absolute_rotation=absolute_rotation)
        elif motion_model == "FORWARD":
            raise ValueError("FORWARD Motion Model is deprecated")
            return self.if_move_by_forward(*params, valid_pose_func)
        else:
            raise ValueError("Unknown motion model %s" % motion_model)

    def if_move_by_axis(self, cur_pose, dpos, dth, valid_pose_func, object_poses=None,
                        absolute_rotation=False):
        """Returns the pose the robot if the robot is moved by the given control.
        The robot is not actually moved underneath;
        There's no check for collision now."""
        robot_pose = [0, 0, 0, 0, 0, 0, 0]

        robot_pose[0] = cur_pose[0] + dpos[0]#max(0, min(, self.width-1))
        robot_pose[1] = cur_pose[1] + dpos[1]#max(0, min(, self.length-1))
        robot_pose[2] = cur_pose[2] + dpos[2]#max(0, min(, self.height-1))

        # Use quaternion
        if not absolute_rotation:
            qx, qy, qz, qw = cur_pose[3:]
        else:
            qx, qy, qz, qw = 0, 0, 0, 1
        R = util.R_quat(qx, qy, qz, qw)
        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = util.R_quat(*util.euler_to_quat(dth[0], dth[1], dth[2]))
            R = R_change * R_prev
        robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6] = R.as_quat()

        if valid_pose_func is not None and valid_pose_func(robot_pose, object_poses=object_poses):
            return tuple(robot_pose)
        else:
            return cur_pose

    # Deprecated! (fix in rotation not applied)
    def if_move_by_forward(self, cur_pose, forward, dth, valid_pose_func):
        robot_facing = self.get_camera_direction(cur_pose, get_tuple=False)
        # project this vector to xy plane, then obtain the "shadow" on xy plane
        forward_vec = robot_facing*forward
        xy_shadow = forward_vec - util.proj(forward_vec, np.array([0,0,1]))
        dy = util.proj(xy_shadow[:2], np.array([0,1]), scalar=True)
        dx = util.proj(xy_shadow[:2], np.array([1,0]), scalar=True)
        yz_shadow = forward_vec - util.proj(forward_vec, np.array([1,0,0]))
        dz = util.proj(yz_shadow[1:], np.array([0,1]), scalar=True)

        dpos = (dx, dy, dz)
        robot_pose = np.array([0,0,0,0,0,0])
        robot_pose[0] = max(0, min(cur_pose[0] + round(dpos[0]), self.width-1))
        robot_pose[1] = max(0, min(cur_pose[1] + round(dpos[1]), self.length-1))
        robot_pose[2] = max(0, min(cur_pose[2] + round(dpos[2]), self.height-1))
        robot_pose[3] = (cur_pose[3] + dth[0]) % 360
        robot_pose[4] = (cur_pose[4] + dth[1]) % 360
        robot_pose[5] = (cur_pose[5] + dth[2]) % 360
        if valid_pose_func(robot_pose):
            return tuple(robot_pose)
        else:
            return cur_pose

    def get_camera_direction(self, cur_pose, get_tuple=True):
        camera_model = self._robot.camera_model
        p, r = camera_model.transform_camera(cur_pose)
        robot_facing = camera_model.get_direction(p)  # p is normalized already
        if get_tuple:
            return tuple(robot_facing)
        else:
            return robot_facing

    def within_view_range(self, robot_pose, point):
        """Returns true if point(x,y,z) is within the field of view of robot
        at the given `robot_pose`; Not considering occlusion"""
        camera_model = self._robot.camera_model
        p, r = camera_model.transform_camera(robot_pose)
        return camera_model.within_range((p,r), list(point) + [1])

    def objects_within_view_range(self, robot_pose, object_poses, volumetric=False):
        """Returns list of ids of the objects within the field of view of robot
        at the given `robot_pose`; Not considering occlusion.

        `volumetric` is True if want to check the entire volume of voxels occupied
        by an object, instead of just checking the voxel at the given object_pose"""
        camera_model = self._robot.camera_model
        p, r = camera_model.transform_camera(robot_pose)
        objects = set({})
        for objid in object_poses:
            if not volumetric:
                if camera_model.within_range((p,r), list(object_poses[objid]) + [1]):
                    objects.add(objid)
            else:
                # Check all voxels that make up this object
                obj = self._objects[objid]
                if isinstance(obj, TetrisObject):
                    cube_poses = obj.cube_poses(*object_poses[objid])
                    for i in range(len(cube_poses)):
                        if camera_model.within_range((p, r), list(cube_poses[i]) + [1]):
                            objects.add(objid)
                            break
        return objects

    def field_of_view_size(self):
        return self._robot.camera_model.field_of_view_size()

    def if_observe_at(self, robot_pose, object_poses, get_poses=False, only_nonempty=False):
        """
        Suppose the robot has pose `robot_pose` and the objects have poses
        given by the dictionary `object_poses` (objid -> (x,y,z)).

        NOTE: DOES NOT RETURN Observation OBJECT.

        If `get_poses` is True,
            Returns the observation as {objid -> [(x,y,z)...]}.
        else:
            Returns the observation as {objid -> [cube_index, ...]}.
        """
        return self.provide_render_observation(robot_pose, object_poses,
                                               get_poses=get_poses,
                                               only_nonempty=only_nonempty)


    def observable(self, objid, robot_pose, object_poses, situation):
        if situation is not None:
            if (situation, objid) in self._observation_cache:
                return self._observation_cache[(situation, objid)]
            else:
                self._observation_cache[(situation, objid)] = False

        camera_model = self._robot.camera_model
        rx, ry, rz, qx, qy, qz, qw = robot_pose
        p, r = camera_model.transform_camera(robot_pose)
        camera_pose = self._robot.camera_pose(robot_pose)

        # forget about the shape of the object; Assume the object is
        # contained within one voxel.
        x,y,z = object_poses[objid]
        if not camera_model.within_range((p, r), (x,y,z,1)):
            # This object is not in the frustum?
            self._observation_cache[(situation, objid)] = False
            return False

        # The object is in the frustum. Now check if it is occluded.
        observable = True
        if self._occlusion_enabled:
            oalt = {}  # observation in parallel coordinates
            for i in object_poses:  # i: object id
                # TODO: this object could be part of the environment (not being searched).
                if i == self._robot_id:
                    continue
                object_pose = object_poses[i]
                considered = True
                if (situation, i) in self._observation_cache:
                    visible = self._observation_cache[(situation, i)]
                    if not visible:
                        considered = False
                if considered:
                    obj_cam = camera_model.perspectiveTransform(object_pose[0],
                                                                object_pose[1],
                                                                object_pose[2],
                                                                camera_pose)
                    xy_key = (round(obj_cam[0], 2), round(obj_cam[1], 2))
                    if xy_key not in oalt:
                        oalt[xy_key] = (i, obj_cam[2]) # objid, cube_depth
                    else:
                        if oalt[xy_key][1] > obj_cam[2]:  # object i is closer
                            if oalt[xy_key][0] == objid:
                                observable = False  # objid is occluded
                            oalt[xy_key] = (i, obj_cam[2])
            # Now make sure all objects still in oalt are marked as observable
            # -- saves future computation.
            for xy_key in oalt:
                self._observation_cache[(situation, oalt[xy_key][0])] = True
        self._observation_cache[(situation, objid)] = observable
        return observable


    def get_frustum_poses(self, robot_pose):
        """Returns a set of voxels poses inside the robot's frustum given
        the robot's pose; Filtererd based on world boundary."""
        return {tuple(v_pose)
                for v_pose in self._robot.camera_model.get_volume(robot_pose)
                if self.in_boundary(v_pose)}

    @property
    def width(self):
        return diff(self._x_range)
    @property
    def length(self):
        return diff(self._y_range)
    @property
    def height(self):
        return diff(self._z_range)
    @property
    def objects(self):
        return self._objects
    @property
    def robot(self):
        return self._robot
    @property
    def robot_id(self):
        return self._robot_id
    # For visualization.
    def render_observation(self, o, object_poses):
        for objid in o:
            glPushMatrix()
            x, y, z = object_poses[objid]
            glTranslatef(x, y, z)
            for content in o[objid]:
                if type(content) == int:
                    i = content
                    self._objects[objid].render_cube(i)
                else:
                    # coords of cube relative to the object pose
                    cx, cy, cz = content
                    cube_coords = (cx-x, cy-y, cz-z)
                    i = self._objects[objid].cube_index(cube_coords)
                    self._objects[objid].render_cube(i)
            glPopMatrix()

    def init_render_search_region(self, ranges):
        xmin, xmax = ranges[0]
        ymin, ymax = ranges[1]
        zmin, zmax = ranges[2]
        vertices = np.array([
            xmin, ymin, zmin,
            xmin, ymin, zmax,
            xmin, ymax, zmin,
            xmin, ymax, zmax,
            xmax, ymin, zmin,
            xmax, ymin, zmax,
            xmax, ymax, zmin,
            xmax, ymax, zmax,
        ])
        indices = np.array([
            0, 1, 3, 2,
            0, 4, 5, 1,
            4, 6, 7, 5,
            2, 6, 7, 3,
            1, 3, 7, 5,
            0, 2, 6, 4
        ])
        colors = np.array([1,1,1,0.3]*8)
        self._search_region_vbos = util.generate_vbo_elements(vertices, indices, colors=colors)

    def render_search_region(self, ranges):
        """
        Visualize the search region given by `ranges`, which is
        [(minx, maxx), (miny, maxy), (minz, maxz)].

        `refresh` is True if want to render a new region.
        """
        # Visualize a box
        if not hasattr(self, "_search_region_vbos"):
            raise ValueError("Have you initialized rendering search region?")
        util.draw_quads(16,
                        self._search_region_vbos[0],  # vertex_vbo
                        self._search_region_vbos[1],  # index_vbo
                        color_vbo=self._search_region_vbos[2],
                        bcolor_vbo=self._search_region_vbos[2],
                        color_size=4)  # color_vbo

    def _indices_yz(self, x=0):
        """Returns the indices of vertices that form a lattice on the y-z plane, given an x coordinate."""
        indices_yz = np.array([[
            x*self._h*self._l+self._h*j+i,   # (y,z)
            x*self._h*self._l+self._h*j+i+1, # (y, z+1)
            x*self._h*self._l+self._h*j+i+self._h+1, # (y+1, z+1)
            x*self._h*self._l+self._h*j+i+self._h # (y+1, z)
        ] for i in range(self._h-1) for j in range(self._l-1)]).flatten()
        return indices_yz

    def _indices_xz(self, y=0):
        """Returns the indices of vertices that form a lattice on the x-z plane, given a y coordinate."""
        indices_xz = np.array([[
            y*self._h+j*self._h*self._l+i,   # (x,z)
            y*self._h+j*self._h*self._l+i+1, # (x,z+1)
            y*self._h+j*self._h*self._l+i+self._h*self._l+1, # (x+1,z+1)
            y*self._h+j*self._h*self._l+i+self._h*self._l, # (x+1,z)
        ] for i in range(self._h-1) for j in range(self._w-1)]).flatten()
        return indices_xz

    def _indices_xy(self, z=0):
        """Returns the indices of vertices that form a lattice on the x-y plane, given a z coordinate."""
        indices_xy = np.array([[
            z+j*self._h*self._l+i*self._h,  # (x,y)
            z+j*self._h*self._l+i*self._h+self._h,  # (x,y+1)
            z+j*self._h*self._l+i*self._h+self._h*self._l+self._h,  # (x+1,y+1)
            z+j*self._h*self._l+i*self._h+self._h*self._l  # (x+1,y)
        ] for i in range(self._l-1) for j in range(self._w-1)]).flatten()
        return indices_xy


    def init_render(self):
        # Vertices
        # has to be y,x,z since that's just how np.meshgrid works.
        yv, xv, zv = np.meshgrid(np.arange(self._l), np.arange(self._w), np.arange(self._h))
        # The resulting vertices are (x,y,z), with z increasing the fastest, y second, then x.
        vertices = np.stack([xv, yv, zv], axis=-1).flatten()

        indices_yz = self._indices_yz(x=0)
        indices_xz = self._indices_xz(y=0)
        indices_xy = self._indices_xy(z=0)
        indices = np.concatenate([indices_xz, indices_yz, indices_xy])
        self._num_indices = len(indices)

        # generate VBO
        self._vertex_vbo, self._index_vbo = util.generate_vbo_elements(vertices, indices)

        object_ids = []# this will be fed into the vertex shader, then the fragment shader.

        for objid in sorted(self._objects):
            obj = self._objects[objid]
            obj.init_render()
            object_ids += obj.object_id_attrb
        self._robot.init_render()

    def cleanup(self):
        glDeleteBuffers(2, np.array([self._vertex_vbo, self._index_vbo]))
        if hasattr(self, "_search_region_vbos"):
            glDeleteBuffers(3, np.array(self._search_region_vbos))
        for objid in self._objects:
            self._objects[objid].cleanup()
        self._robot.cleanup()
        print("Bye")

    def render(self, robot_pose, object_poses, gridlines=True, render_fov=False):
        if gridlines:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBindBuffer(GL_ARRAY_BUFFER, self._vertex_vbo);
            glVertexPointer(3, GL_FLOAT, 0, None);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_vbo);
            glDrawElements(GL_QUADS, 4*self._num_indices, GL_UNSIGNED_INT, None)

        # Render objects

        for objid in sorted(self._objects):
            obj = self._objects[objid]
            glPushMatrix()
            x, y, z = object_poses[objid]
            glTranslatef(x, y, z)
            obj.render()
            glPopMatrix()

        # Render the robot
        glPushMatrix()

        rx, ry, rz, qx, qy, qz, qw = robot_pose
        thx, thy, thz = util.quat_to_euler(qx, qy, qz, qw)
        glTranslatef(rx, ry, rz)
        glTranslatef(0.5, 0.5, 0.5)  # align the cube with the grid lines; 0.5 is half of cube length.

        glRotatef(thz, 0, 0, 1)
        glRotatef(thy, 0, 1, 0)
        glRotatef(thx, 1, 0, 0)

        self._robot.render(render_fov=render_fov)
        glPopMatrix()


    ## I WILL NOT TOUCH THIS.
    def provide_render_observation(self, robot_pose, object_poses,
                                   get_poses=False, only_nonempty=False, return_oalt=False):
        """Same method as `if_observe_at`"""
        # Since the camera_model doesn't know its position in the world,
        # the world needs to obtain a configuration of the camera as if
        # it is moved to where the robot currently is, and obtain observation
        # from that configuration. The camera's job is to tell if something
        # is visible.
        camera_model = self._robot.camera_model
        rx, ry, rz, qx, qy, qz, qw = robot_pose
        p, r = camera_model.transform_camera(robot_pose)
        camera_pose = self._robot.camera_pose(robot_pose)

        o = {}
        oalt = {}  # observation in parallel coordinates
        for objid in object_poses:
            o[objid] = set({})
            obj = self._objects[objid]
            x, y, z = object_poses[objid]
            if isinstance(obj, TetrisObject):
                cube_poses = obj.cube_poses(x, y, z)
                for i in range(len(cube_poses)):
                    if camera_model.within_range((p, r), list(cube_poses[i]) + [1]):
                        if self._occlusion_enabled:
                            # if the cube pose is in "hidden", then it will never be
                            # part of the observation.
                            if tuple(cube_poses[i]) in self._hidden:
                                continue
                            cube_cam = camera_model.perspectiveTransform(cube_poses[i][0],
                                                                         cube_poses[i][1],
                                                                         cube_poses[i][2],
                                                                         camera_pose)
                            xy_key = (round(cube_cam[0], 2), round(cube_cam[1], 2))
                            if(xy_key not in oalt.keys()):
                                # primitive of oalt: { (x,y) : objid, cube_index, cube_depth, cube_pose }
                                oalt[xy_key] = (objid, i, cube_cam[2], tuple(cube_poses[i]))
                                if get_poses:
                                    o[objid].add(tuple(cube_poses[i]))  # the pose of cube is added
                                else:
                                    o[objid].add(i)  # the index of the cube is added
                            else:
                                # another cube at the same viewing ray -> compare depth for occulusion
                                if oalt[xy_key][2] > cube_cam[2]: # since we are facing negative z-direction
                                    prev_objid = oalt[xy_key][0]
                                    if get_poses:
                                        o[prev_objid].remove(oalt[xy_key][3])
                                        o[objid].add(tuple(cube_poses[i]))  # the pose of cube is added
                                    else:
                                        o[prev_objid].remove(oalt[xy_key][1])
                                        o[objid].add(i)  # the index of the cube is added
                                    oalt[xy_key] = (objid, i, cube_cam[2], tuple(cube_poses[i]))
                        else:
                            # no occlusion checking
                            if get_poses:
                                o[objid].add(tuple(cube_poses[i]))  # the pose of cube is added
                            else:
                                o[objid].add(i)  # the index of the cube is added
        if only_nonempty:
            o = {o[objid]
                 for objid in o
                 if len(o[objid]) > 0}
        if return_oalt:
            return o, oalt
        else:
            return o

    # This is too computationally expensive for planning
    def provide_observation(self, robot_pose, object_poses, only_nonempty=False,
                            alpha=1000., beta=0., log=False):
        """Same method as `if_observe_at`; Take into account uncertainty
        in the robot's observation model."""
        # Since the camera_model doesn't know its position in the world,
        # the world needs to obtain a configuration of the camera as if
        # it is moved to where the robot currently is, and obtain observation
        # from that configuration. The camera's job is to tell if something
        # is visible.

        # Compute camera pose
        camera_model = self._robot.camera_model
        rx, ry, rz, qx, qy, qz, qw = robot_pose
        p, r = camera_model.transform_camera(robot_pose)
        camera_pose = self._robot.camera_pose(robot_pose)

        # Obtain volumetric observation
        o, oalt = self.provide_render_observation(robot_pose, object_poses,
                                                  get_poses=True,
                                                  only_nonempty=only_nonempty,
                                                  return_oalt=True)
        voxel_to_objid = {}
        for objid in o:
            for voxel_pose in o[objid]:
                voxel_to_objid[voxel_pose] = objid

        volume = self._robot.camera_model.get_volume(robot_pose)
        voxels = {}  # voxels according to the definition in observation
        for x,y,z in volume:
            # Need to make sure the voxel is valid in the
            # and that voxel is not forced to be hidden
            if not self.in_boundary((x,y,z)) or (x,y,z) in self._hidden:
                continue # skip
            if(x,y,z) in voxel_to_objid:
                voxel = Voxel((x,y,z), voxel_to_objid[(x,y,z)])
            else:
                valt = camera_model.perspectiveTransform(x, y, z, camera_pose)
                xy_key = (round(valt[0], 2), round(valt[1], 2))
                if(xy_key not in oalt.keys()):
                    # Free
                    voxel = Voxel((x,y,z), Voxel.FREE)
                else:
                    if oalt[xy_key][2] > valt[2]:  # voxel x,y,z is closer to the camera than the obstacle
                        voxel = Voxel((x,y,z), Voxel.FREE)
                    else:
                        voxel = Voxel((x,y,z), Voxel.UNKNOWN)
            voxels[(x,y,z)] = voxel
        return OOObservation(voxels, OOObservation.T_VOLUME)
