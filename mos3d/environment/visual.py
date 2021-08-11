import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import *

import numpy as np

from mos3d.models.world.objects import *
from mos3d.models.world.robot import Robot
from mos3d.models.world.world import GridWorld
from mos3d.oopomdp import TargetObjectState, RobotState, MOTION_ACTION_NAMES, MOTION_ACTIONS,\
    Actions, SimLookAction, LookAction, DetectAction, SimMotionAction, MOTION_MODEL
from mos3d.environment.env import parse_worldstr, Mos3DEnvironment
from mos3d.models.observation\
    import OOObservation, ObjectObservationModel, VoxelObservationModel, M3ObservationModel
from mos3d.models.transition import M3TransitionModel
from mos3d.models.reward import GoalRewardModel, GuidedRewardModel
import mos3d.util as util
from pomdp_py import Environment, ObjectState, OOState
# from pomdp_py import Environment, OOPOMDP_ObjectState, BeliefState, OOPOMDP_State, OOPOMCP_Histogram


class Mos3DViz:

    """
    Visualization of a Mos3D World, as a pygame.
    Allows user to control the robot, if controllable=True.
    """

    def __init__(self, env, gridworld, fps=30, controllable=True, show_robot_view=False):
        self._env = env
        self._gridworld = gridworld
        self._object_poses = env.object_poses  # objects are static; get this dictionary beforehand.
        self._fps = fps
        self._playtime = 0
        self._width = 500
        self._height = self._width
        self._controllable = controllable   # True if the user can control the robot
        self._motion_model = MOTION_MODEL
        self._display_surf = None
        self._show_robot_view = show_robot_view
        self._running = False
        self._rerender = False  # render again
        self._last_observation = None
        self._last_real_action = None
        self._last_search_region = (None, False)  # ranges, refresh
        self._info = None # info to show on window's title

    @property
    def gridworld(self):
        return self._gridworld

    @property
    def last_real_action(self):
        return self._last_real_action

    @property
    def motion_model(self):
        return self._motion_model

    def on_init(self):
        self._running = True

        pygame.init()

        if self._show_robot_view:
            window_width = self._width*2
            window_height = self._height*2
        else:
            window_width = self._width*2
            window_height = self._height

        self._display_surf = pygame.display.set_mode((window_width,
                                                      window_height),
                                                      DOUBLEBUF | OPENGL)
        self._init_observation_viewport()
        if self._show_robot_view:
            self._init_camera_viewport()
        self._init_main_viewport()

        self._gridworld.init_render()

        glEnable(GL_DEPTH_TEST)  # order object according to depth
        glEnable(GL_BLEND)  # enable alpha
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # vertices for axes
        vertices = np.array([0,0,0,
                             0,0,0,
                             0,0,0,
                             2*self._gridworld.width,0,0,
                             0,2*self._gridworld.length,0,
                             0,0,2*self._gridworld.height])
        colors = np.array([1,0,0,  # origin - red
                           0,1,0,  # origin - green
                           0,0,1,  # origin - blue
                           1,0,0,  # Red
                           0,1,0,  # Green
                           0,0,1]) # Blue
        indices = np.array([0,3,1,4,2,5])
        self._axes_vertex_vbo, self._axes_index_vbo, self._axes_color_vbo \
            = util.generate_vbo_elements(vertices, indices, colors)

        self._clock = pygame.time.Clock()
        self._rerender = True  # render again
        self._info = {"fps": "",
                      "playtime": self._playtime,
                      "action": "",
                      "resolution": -1}
        return True


    def _Test_observation_factoring(self, action):
        # provide observation
        if isinstance(action, LookAction):
            o = self._gridworld.provide_render_observation(self._env.robot_pose,
                                                           self._object_poses)
            self._last_observation = o

            # THIS IS A TEST; ACTUAL OBSERVATION; TEST FACTORING AND MERGING
            true_o = self._gridworld.provide_observation(self._env.robot_pose, self._object_poses)
            factored = true_o.factor(self._env.state, None)
            volume = self._gridworld.robot.camera_model.get_volume(self._env.robot_pose)
            filtered_volume = {tuple(v) for v in volume if self._gridworld.in_boundary(v)}
            merged = M3ObservationModel.merge_observations(factored, self._env.state, action, filtered_volume)
            if merged == true_o:
                print("True")
            else:
                print(len(merged.voxels), len(true_o.voxels))
                print({p for p in true_o.voxels if true_o.voxels[p] != merged.voxels[p]})

    def save_frame(self, path):
        pygame.image.save(self._display_surf, path)

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

        elif event.type == pygame.KEYDOWN:
            self._rerender = True  # render again
            if not self._handle_move_camera_event(event):
                # A test; can comment out.
                action = None

                a = self._handle_control_robot_event(event)
                if a is not None:
                    # motion action
                    action = SimMotionAction(a, motion_model=self._motion_model)
                    self._last_real_action = action
                    # self._env.move_robot(self._env.robot_pose, u[0], u[1],
                    #                      motion_model=self._motion_model)
                else:
                    if event.key == pygame.K_SPACE:
                        print("DETECT!")
                        action = DetectAction()
                    else:
                        if self._motion_model == "AXIS":
                            if event.key == pygame.K_l:
                                print("Look!")
                                action = LookAction()
                        elif self._motion_model == "TRANS":
                            a = self._handle_directional_look_action(event)
                            if a is not None:
                                action = SimLookAction(a)
                                print(action)
                    self._last_real_action = action
                if action is not None:
                    if self._controllable:
                        self._env.state_transition(action, execute=True)
                        self._Test_observation_factoring(action)
                return action

    def rendering_fov(self):
        return isinstance(self._last_real_action, LookAction)\
            or isinstance(self._last_real_action, DetectAction)

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
        if self.rendering_fov():
            o = self._gridworld.provide_render_observation(self._env.robot_pose,
                                                           self._object_poses)
            self._last_observation = o
        else:
            self._last_observation = None


    def on_render(self, rerender=None):
        self._info["playtime"] = self._playtime
        self._info["fps"] = self._clock.get_fps()
        #    Detected Objects: %s"\
        text = "FPS: %.2f   Playtime: %.2f    Action: %s    Resolution: %d"\
            % (self._info["fps"], self._info["playtime"], self._info["action"],
               self._info["resolution"])#, self._info["detected_objects"])
        pygame.display.set_caption(text)

        if rerender is not None:
            self._rerender = rerender

        if self._rerender:
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glEnableClientState(GL_VERTEX_ARRAY)

            # Main viewport
            self._set_viewport(0)
            self._look_at_scene(0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self._gridworld.render(
                self._env.robot_pose,
                self._object_poses,
                render_fov=self.rendering_fov()) # Draw the gridworld
            self._render_axes()
            self._render_search_region()

            # Observation viewport
            self._set_viewport(1)
            self._look_at_scene(1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self._render_axes()
            self._render_observation()

            # Robot camera viewport
            self._set_viewport(2)
            self._look_from_robot()
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self._gridworld.render(self._env.robot_pose,
                                   self._object_poses, gridlines=False) # Draw the gridworld

            glDisableClientState(GL_VERTEX_ARRAY)
            self._rerender = False
        pygame.display.flip()
        pygame.time.wait(10)

    def on_cleanup(self):
        self._gridworld.cleanup()
        glDeleteBuffers(3, np.array([self._axes_vertex_vbo, self._axes_index_vbo, self._axes_color_vbo]))
        pygame.display.quit()
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            for event in pygame.event.get():
                action = self.on_event(event)
                yield action
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def update_search_region(self, ranges):
        self._last_search_region = (ranges, True)

    def update_info(self, real_action=None, resolution=None):
        if real_action is None:
            real_action = self._last_real_action
        if resolution is None:
            resolution = 1
        action_text = str(real_action) if real_action not in MOTION_ACTION_NAMES[self._motion_model]\
            else MOTION_ACTION_NAMES[self._motion_model][real_action]
        self._info["action"] = action_text
        self._info["resolution"] = resolution

    def update_observation(self, renderable_observation):
        self._last_observation = renderable_observation

    def update(self, real_action, robot_pose, object_poses, observation=None):
        if observation is None:
            observation = self._gridworld.provide_render_observation(robot_pose,
                                                                     object_poses)
        self._last_real_action = real_action
        self._last_observation = observation  # must be renderable

    def _handle_move_camera_event(self, event):
        result = True
        if event.key == pygame.K_LEFT:
            self._viewport0_rotz -= 5
        elif event.key == pygame.K_RIGHT:
            self._viewport0_rotz += 5
        elif event.key == pygame.K_UP:
            self._viewport0_rotp -= 5
        elif event.key == pygame.K_DOWN:
            self._viewport0_rotp += 5
        else:
            result = False
        return result

    def _handle_directional_look_action(self, event):
        a = None
        if self._motion_model == "TRANS":
            if event.key == pygame.K_k:
                a = "look+thx"
            elif event.key == pygame.K_j:
                a = "look-thx"
            elif event.key == pygame.K_h:
                a = "look+thy"
            elif event.key == pygame.K_l:
                a = "look-thy"
            elif event.key == pygame.K_u:
                a = "look+thz"
            elif event.key == pygame.K_m:
                a = "look-thz"
        return a

    def _handle_control_robot_event(self, event):
        a = None
        # Control robot
        # AXIS MOTION_MODEL
        if self._motion_model == "TRANS":
            if event.key == pygame.K_s:
                a = 0
            elif event.key == pygame.K_d:
                a = 1
            elif event.key == pygame.K_a:
                a = 2
            elif event.key == pygame.K_f:
                a = 3
            elif event.key == pygame.K_x:
                a = 4
            elif event.key == pygame.K_w:
                a = 5
        if self._motion_model == "AXIS":
            if event.key == pygame.K_s:
                a = 0
            elif event.key == pygame.K_d:
                a = 1
            elif event.key == pygame.K_a:
                a = 2
            elif event.key == pygame.K_f:
                a = 3
            elif event.key == pygame.K_x:
                a = 4
            elif event.key == pygame.K_w:
                a = 5
            elif event.key == pygame.K_e:
                a = 6
            elif event.key == pygame.K_c:
                a = 7
            elif event.key == pygame.K_r:
                a = 8
            elif event.key == pygame.K_v:
                a = 9
            elif event.key == pygame.K_t:
                a = 10
            elif event.key == pygame.K_b:
                a = 11
        elif self._motion_model == "FORWARD":
            # FORWARD/BACKWORD motion model
            if event.key == pygame.K_w:
                a = 0
            elif event.key == pygame.K_s:
                a = 1
            elif event.key == pygame.K_e:
                a = 2
            elif event.key == pygame.K_c:
                a = 3
            elif event.key == pygame.K_r:
                a = 4
            elif event.key == pygame.K_v:
                a = 5
            elif event.key == pygame.K_t:
                a = 6
            elif event.key == pygame.K_b:
                a = 7
        return a

    def _look_at_scene(self, viewport, orthographic=False):
        if orthographic:
            util.apply_orthographic_transform(-15, 15, -15, 15,
                                              0.1,
                                              max(self._gridworld.width,
                                                  self._gridworld.length,
                                                  self._gridworld.height)*5)
        else:
            util.apply_perspective_transform(45,
                                             self._width/self._height,
                                             0.1,
                                             max(self._gridworld.width,
                                                 self._gridworld.length,
                                                 self._gridworld.height)*5)
        glTranslatef(0, 0, -min(self._gridworld.width,
                                self._gridworld.length,
                                self._gridworld.height)*3)
        # look at the origin in a natural way
        glRotatef(-90,0,0,1)
        glRotatef(-90,0,1,0)
        glRotatef(-45,0,0,1)
        glRotatef(-30,1,-1,0)
        glTranslatef(0, 0, -self._gridworld.height/10)

        # rotate the camera view by saved amount
        if viewport == 0:
            glRotatef(self._viewport0_rotz, 0, 0, 1)
            glRotatef(self._viewport0_rotp, 1, -1, 0)


    def _look_from_robot(self):
        """Note: This is merely a shift in camera view, not a honest display of the
        observation that the robot receives.

        Warning: This method does not work properly."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity();
        camera_model = self._gridworld.robot.camera_model
        # gluPerspective(45,
        #                self._width/self._height,
        #                0.1,
        #                max(self._gridworld.width,
        #                    self._gridworld.length,
        #                    self._gridworld.height)*5)
        gluPerspective(camera_model.fov,
                       camera_model.aspect_ratio,
                       camera_model.near,
                       camera_model.far+0.1)
        glRotatef(-90, 0, 0, 1)

        camera_pose = self._gridworld.robot.camera_pose(self._env.robot_pose)
        rx, ry, rz, qx, qy, qz, qw = camera_pose
        thx, thy, thz = util.quat_to_euler(qx, qy, qz, qw)
        glRotatef(-thz, 0, 0, 1)
        glRotatef(-thy, 0, 1, 0)
        glRotatef(-thx, 1, 0, 0)
        glTranslatef(-rx-0.5,
                     -ry-0.5,
                     -rz-0.5)

    def _set_viewport(self, num):
        if num == 0:
            util.set_viewport(0, 0, self._width, self._height)
        elif num == 1:
            util.set_viewport(self._width, 0, self._width, self._height)
        # if num == 0:
        #     util.set_viewport(0, self._height, self._width, self._height)
        # elif num == 1:
        #     util.set_viewport(self._width, self._height, self._width, self._height)
        elif num == 2:
            util.set_viewport(0, self._height, self._width, self._height)
        # elif num == 3:
        #     util.set_viewport(self._width, 0, self._width, self._height)
        else:
            raise ValueError("Invalid viewport %d" % num)

    def _init_main_viewport(self):
        # viewport (drawing the scene)
        self._set_viewport(0)
        self._camera = "scene"  # either scene or robot
        # Projection matrix
        self._viewport0_rotz = 0
        self._viewport0_rotp = 0
        # Modelview matrix - everything afterwards is about modelview (unless camera changes)
        glMatrixMode(GL_MODELVIEW)

    def _init_observation_viewport(self):
        # viewport (drawing the scene)
        self._set_viewport(1)
        self._viewport1_rotz = 0
        self._viewport1_rotp = 0
        # Modelview matrix - everything afterwards is about modelview (unless camera changes)
        glMatrixMode(GL_MODELVIEW)

    def _init_camera_viewport(self):
        # viewport (drawing the scene)
        self._set_viewport(2)
        self._viewport2_rotz = 0
        self._viewport2_rotp = 0
        # Modelview matrix - everything afterwards is about modelview (unless camera changes)
        glMatrixMode(GL_MODELVIEW)

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

    def _render_observation(self):
        if self._last_observation is not None:
            self._gridworld.render_observation(self._last_observation, self._object_poses)

    def _render_search_region(self):
        ranges, refresh = self._last_search_region
        if refresh:
            self._gridworld.init_render_search_region(ranges)
            self._last_search_region = (ranges, False)
        if ranges is not None:
            self._gridworld.render_search_region(ranges)

# ----------- Testing -------------
world1 =\
"""
10
10
10

orange_ricky 2 2 0
hero 0 4 0
teewee 2 4 3
teewee 5 6 5
smashboy 3 3 4
smashboy 6 6 0
cube 9 5 9
teewee 2 5 9
cube 8 5 0
cube 0 0 9
hero 0 2 5
cube 5 9 0
cube 7 8 0
---
robot 8 1 0 0 0 0 occlusion 45 1.0 1.0 10
"""

world_simple =\
"""
5
5
5

teewee 1 0 1
cube 0 0 3
---
robot 4 0 0 0 0 0 frustum 45 1.0 1 5
"""

world_basic =\
"""
3
3
3

cube 0 0 0
cube 1 0 0
cube 0 0 1
cube 0 1 0
---
robot 2 0 0 0 0 0 occlusion 45 1.0 0.1 4
"""

world_trivial =\
"""
2
2
2

cube 0 1 0
---
robot 0 0 1 0 0 0 occlusion 45 1.0 0.1 5
"""

world_occlusion =\
"""
4
4
4

cube 0 0 0
cube 1 0 0
cube 0 0 1
cube 0 1 0 obstacle
---
robot 2 0 0 0 0 0 occlusion 45 1.0 0.1 4
"""

worldocc2=\
"""
4
4
4

cube 0 0 0 hidden
cube 1 0 0 obstacle
cube 0 1 0 obstacle
cube 0 0 1 obstacle
---
robot 3 1 0 0 0 0 occlusion 45 1.0 0.1 4
"""

if __name__ == "__main__":
    gridworld, init_state = parse_worldstr(worldocc2)

    # print("** TEST Observation Model")
    # oo = VoxelObservationModel(1, gridworld, observe_when_look=False)
    # objo = oo.sample(init_state, None)
    # # for vp in objo.voxels:
    # #     if objo.voxels[vp].label == 1:
    # #         print(vp)
    # print(init_state.object_poses[1])
    # print(objo)
    # print(oo.probability(objo, init_state, None))

    # oo = M3ObservationModel(gridworld, voxel_model=True)
    # objo = oo.sample(init_state, None)
    # print(objo)

    camera_pose = gridworld._robot.camera_pose(init_state.robot_pose)
    print(gridworld._robot.camera_model.perspectiveTransform(0,0,0, camera_pose))#(0,0,0,0,0,0,1)))#camera_pose))

    T = M3TransitionModel(gridworld)
    R = GoalRewardModel(gridworld)

    env = Mos3DEnvironment(init_state, gridworld, T, R)
    env = Mos3DViz(env, gridworld, fps=15)
    for action in env.on_execute():
        pass
