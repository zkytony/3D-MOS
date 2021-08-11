import pomdp_py
from mos3d.models.observation import OOObservation, Voxel
from mos3d.oopomdp import LookAction, MotionAction

class M3Agent(pomdp_py.Agent):
    def __init__(self,
                 gridworld,
                 init_belief,
                 policy_model,
                 transition_model,
                 observation_model,
                 reward_model,
                 name="M3Agent"):
        self._gridworld = gridworld
        self.name = name
        self._explored_voxels = set({})
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def clear_history(self):
        self._history = None

    def exploration_ratio(self, real_observation):
        """Returns the ratio of explored voxels, given real_observation
        and previously explored voxels."""
        self._explored_voxels = self._explored_voxels | set(real_observation.voxels.keys())
        world_size = self._gridworld.width*self._gridworld.length*self._gridworld.height
        return len(self._explored_voxels) / world_size


    def convert_real_observation_to_planning_observation(self, real_observation, real_action):
        """This function should be called BEFORE belief update (because for POMCP, the belief
        update happens only when updating the MCTS tree, which requires the observation in
        the right format to be ready.

        Convert a real observation, which is a set of voxels
        each could be labeled as 1,..,N, or FREE or UNKNOWN, to planning
        observation, which is (robot_pose, o1, ... oN).

        For planing, we are using "VoxelObjectObseravtion", that is, each
        oi should just contain one voxel; Technically, the agent has no
        idea where the objects are in groundtruth, so it has no idea which
        location each oi should be located at.

        We do want to produce (robot_pose, o1, ... oN) because this is the
        format of observation used for planning (more efficient). In planning,
        there's always a sampled state, so oi will just be located at si.

        Indeed, given a real observation, the object i is either observed, or not.
        So, oi can be labeled either i, or OTHER (=FREE in paper). When the label
        is i, there is at least one voxel in real_observation labeled by i. So
        we can assign oi to be located at one of these voxels labeled by i. If
        OTHER, then we have to come up with a location to set the oi. A reasonable
        choice is to use the MPE of the current belief.

        Note that the converted observation will not be used for belief update.
        It has the same format as the planning observation, so we supposedly can
        use it to truncate the MCTS tree, which is what it is used for.

        .... or.... we just replan. Throw out the previous tree.
        """
        if not isinstance(real_action, LookAction):
            # No observation received. So the observation is some fixed value (see observation.py)
            return OOObservation({}, None)
        else:
            # Obtain the right robot pose (real_action has been executed)
            # Need to apply the real_action to update robot belief
            # mpe_state = self.cur_belief.mpe()
            # if isinstance(real_action, pomdp_py.Option):
            #     while not real_action.terminate(mpe_state):
            #         action = real_action.sample(mpe_state)
            #         mpe_state = self.transition_model(mpe_state, action)
            # next_mpe_state = self.transition_model.sample(mpe_state, real_action)
            # robot_pose = next_mpe_state.robot_pose

            voxel_observations = {}  # map from objid to Voxel
            # First collect the MPE pose for each object
            for objid in self._gridworld.target_objects:
                if isinstance(self.cur_belief, pomdp_py.OOBelief):
                    belief_obj = self.cur_belief.object_belief(objid)
                    obj_mpe_pose = belief_obj.mpe()['pose']
                else:
                    obj_mpe_pose = self.cur_belief.mpe().object_poses[objid]
                voxel_observations[objid] = Voxel(obj_mpe_pose, Voxel.UNKNOWN)

            # Then, if the MPE pose is observed, labeled it according to observation.
            for objid in voxel_observations:
                voxel_pose = voxel_observations[objid].pose
                if voxel_pose in real_observation.voxels:
                    observed_voxel = real_observation.voxels[voxel_pose]
                    if observed_voxel.label != Voxel.UNKNOWN:
                        if observed_voxel.label == objid:
                            voxel_observations[objid].label = objid
                        else:
                            voxel_observations[objid].label = Voxel.OTHER
                    # If label is UNKNOWN, then the voxel is actually not in FOV;
                    # So keep the original label with is UNKNOWN

            # Finally, override the voxel observations using real observation.
            for voxel_pose in real_observation.voxels:
                # Iteration over the voxels is random; If an object
                # occupies multiple voxels, a random voxel is chosen to represent
                # that object.
                voxel = real_observation.voxels[voxel_pose]
                if type(voxel.label) == int:
                    voxel_observations[voxel.label] = voxel#ObjectVoxelObservation(voxel.label, voxel)

            # Return observation for planning
            return OOObservation(voxel_observations,
                                 OOObservation.T_VOXEL)
