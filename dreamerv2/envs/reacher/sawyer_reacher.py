from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
from gym.envs.mujoco import mujoco_env
import os

# import gin

SCRIPT_DIR = os.path.dirname(__file__)


class SawyerReachingEnv(mujoco_env.MujocoEnv):

    """
    Reaching to a desired end-effector position while controlling the 7 joints of sawyer
    """

    def __init__(
        self,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        reward_scaling=1.0,
        use_force=False,
        render_field=False
    ):
        # starting positions
        self.start_positions = []
        self.start_positions.append(
            np.array(
                [
                    -1.44397e-06,
                    -0.832489,
                    0.0299997,
                    1.68,
                    0.0580013,
                    -0.232,
                    -2.16526e-07,
                ]
            )
        )
        self.add_field = use_force
        self.render_field = render_field

        if self.add_field:

            self.attract_force_num = 4
            self.repul_force_num = 4


            self.force_source_range = Box(
                low=np.array([0.7, -0.35, 0.25]), high=np.array([0.95, 0.35, 0.85])
            )

            self.force_field_range = Box(
                low=np.array(0.05), high=np.array(0.5)
            )

            self.is_field_obs = False

            '''
            # force field
            self.attract_forces = [self.force_source_range.sample() for _ in range(self.attract_force_num)]
            self.attract_force_range = [self.force_field_range.sample() for _ in range(self.attract_force_num)]

            self.repul_forces = [self.force_source_range.sample() for _ in range(self.repul_force_num)]
            self.repul_force_range = [self.force_field_range.sample() for _ in range(self.repul_force_num)]

            '''
            # force field
            self.attract_forces = [[0.9, -0.05, 0.65]]
            self.attract_force_range = [0.28]

            self.repul_forces = [[0.9, -0.15, 0.43],[0.9, 0.09, 0.62]]
            self.repul_force_range = [0.13, 0.2]

        # vars
        self.reward_scaling = reward_scaling
        self.action_mode = action_mode
        self.num_joint_dof = 7
        self.frame_skip = 100
        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, "assets/sawyer_reach.xml")
        if goal_site_name is None:
            goal_site_name = "goal_reach_site"
        self.body_id_ee = 0
        self.site_id_ee = 0
        self.site_id_goal = 0
        self.is_eval_env = False

        # Sparse reward setting
        self.truncation_dist = 0.15
        self.sparse_margin = 1

        # create the env
        self.startup = True
        mujoco_env.MujocoEnv.__init__(
            self, xml_path, self.frame_skip
        )  # self.model.opt.timestep is 0.0025 (w/o frameskip)
        self.startup = False

        # initial position of joints
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = np.zeros(len(self.data.qvel))

        # joint limits
        self.limits_lows_joint_pos = self.model.actuator_ctrlrange.copy()[:, 0]
        self.limits_highs_joint_pos = self.model.actuator_ctrlrange.copy()[:, 1]

        # set the action space (always between -1 and 1 for this env)
        self.action_highs = np.ones((self.num_joint_dof,))
        self.action_lows = -1 * np.ones((self.num_joint_dof,))
        self.action_space = Box(low=self.action_lows, high=self.action_highs)

        # set the observation space
        obs_size = self.get_obs_dim()
        self.observation_space = Box(
            low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf
        )

        # vel limits
        joint_vel_lim = 0.04  # magnitude of movement allowed within a dt [deg/dt]
        self.limits_lows_joint_vel = -np.array([joint_vel_lim] * self.num_joint_dof)
        self.limits_highs_joint_vel = np.array([joint_vel_lim] * self.num_joint_dof)

        # ranges
        self.action_range = self.action_highs - self.action_lows
        self.joint_pos_range = self.limits_highs_joint_pos - self.limits_lows_joint_pos
        self.joint_vel_range = self.limits_highs_joint_vel - self.limits_lows_joint_vel

        # ids
        self.body_id_ee = self.model.body_names.index("end_effector")
        self.site_id_ee = self.model.site_name2id("ee_site")
        self.site_id_goal = self.model.site_name2id(goal_site_name)

    def override_action_mode(self, action_mode):
        self.action_mode = action_mode

    def get_obs_dim(self):
        return len(self.get_obs())

    def get_obs(self):
        """ state observation is joint angles + joint velocities + ee pose """
        angles = self._get_joint_angles()
        velocities = self._get_joint_velocities()
        ee_pose = self._get_ee_pose()
        return np.concatenate([angles, velocities, ee_pose])

    def _get_joint_angles(self):
        return self.data.qpos.copy()

    def _get_joint_velocities(self):
        return self.data.qvel.copy()

    def _get_ee_pose(self):
        """ ee pose is xyz position + orientation quaternion """
        return self.data.site_xpos[self.site_id_ee].copy()

    def reset_model(self):
        angles_idx = np.random.randint(len(self.start_positions))
        angles = self.start_positions[angles_idx]

        velocities = self.init_qvel.copy()
        self.set_state(
            angles, velocities
        )  # this sets qpos and qvel + calls sim.forward

        return self.get_obs()

    def do_step(self, action):
        if self.startup:
            feasible_desired_position = 0 * action
        else:
            # clip to action limits
            action = np.clip(action, self.action_lows, self.action_highs)

            # get current position
            curr_position = self._get_joint_angles()

            if self.action_mode == "joint_position":
                # scale incoming (-1,1) to self.joint_limits
                desired_position = (
                    ((action - self.action_lows) * self.joint_pos_range)
                    / self.action_range
                ) + self.limits_lows_joint_pos

                # make the
                feasible_desired_position = self.make_feasible(
                    curr_position, desired_position
                )

            elif self.action_mode == "joint_delta_position":
                # scale incoming (-1,1) to self.vel_limits
                desired_delta_position = (
                    ((action - self.action_lows) * self.joint_vel_range)
                    / self.action_range
                ) + self.limits_lows_joint_vel

                # add delta
                feasible_desired_position = curr_position + desired_delta_position

        self.do_simulation(feasible_desired_position, self.frame_skip)

    def step(self, action):
        """ apply the 7DoF action provided by the policy """
        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
        done = False
        # info = np.array([score, 0, 0, 0, 0])  # can populate with more info, as desired, for tb logging
        info = {"score": score}
        reward /= self.reward_scaling
        return obs, reward, done, info

    def make_feasible(self, curr_position, desired_position):
        # compare the implied vel to the max vel allowed
        max_vel = self.limits_highs_joint_vel
        implied_vel = np.abs(desired_position - curr_position)

        # limit the vel
        actual_vel = np.min([implied_vel, max_vel], axis=0)

        # find the actual position, based on this vel
        sign = np.sign(desired_position - curr_position)
        actual_difference = sign * actual_vel
        feasible_position = curr_position + actual_difference

        return feasible_position

    def compute_reward(self, get_score=False, goal_id_override=None, button=None):

        # get goal id
        if goal_id_override is None:
            goal_id = self.site_id_goal
        else:
            goal_id = goal_id_override

        # get coordinates of the sites in the world frame
        ee_xyz = self.data.site_xpos[self.site_id_ee].copy()
        goal_xyz = self.data.site_xpos[goal_id].copy()

        # score
        score = -np.linalg.norm(ee_xyz - goal_xyz)

        ### DISTANCE
        dist = 5 * np.linalg.norm(ee_xyz - goal_xyz)

        offset = -(
            -(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5))
        )
        reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5)) + offset

        if dist < self.truncation_dist:
            sparse_reward = reward + self.sparse_margin  # create a gap of 1
        else:
            sparse_reward = 0

        def compute_field_reward(ee_xyz, sources, field_ranges, is_attractive):
            field_reward = 0
            for i in range(len(sources)):
                dist = np.linalg.norm(ee_xyz - sources[i])
                if dist <= field_ranges[i]:
                    field_reward += is_attractive * (1/((2 + dist)**2))
            return field_reward

        if self.add_field:
            attractive_field_reward = compute_field_reward(ee_xyz, self.attract_forces, self.attract_force_range, 10)
            repulsive_field_reward = compute_field_reward(ee_xyz, self.repul_forces, self.repul_force_range, -20)
            reward += (attractive_field_reward + repulsive_field_reward)

        if get_score:
            return reward, score, sparse_reward
        else:
            return reward

    def viewer_setup(self):
        # side view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0.4
        # self.viewer.cam.lookat[0] = 0.75
        # self.viewer.cam.lookat[1] = 0.1
        # self.viewer.cam.lookat[2] = 0.4
        # self.viewer.cam.distance = 0.2
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 220
        self.viewer.cam.distance = 0.9
        self.viewer.cam.lookat[1] = 0.1
        self.viewer.cam.lookat[0] = 0.75
        self.viewer.cam.lookat[2] = 0.4

        # self.viewer.cam.trackbodyid = -1

    def reset(self):
        # reset task (this is a single-task case)
        self.model.site_pos[self.site_id_goal] = np.array([0.7, 0.2, 0.4])

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def goal_visibility(self, visible):

        """ Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't """

        if visible:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 1])
        else:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 0])

    def render(self, mode="human", width=500, height=500):
        if self.add_field and self.render_field:
            self.render_force_field(self.attract_forces, self.attract_force_range, self.repul_forces,
                                    self.repul_force_range)
        if mode == "rgb_array":
            # self._get_viewer(mode='rgb_array').render()
            # window size used for old mujoco-py:
            # width, height = 500, 500
            self._get_viewer(mode="rgb_array").render(width, height)
            data = self._get_viewer(mode="rgb_array").read_pixels(
                width, height, depth=False
            )
            return np.flipud(data)
        elif mode == "human":
            self._get_viewer(mode="human").render()

    def render_force_field(self, attractive_forces, attractive_force_ranges, repulsive_forces, repulsive_force_ranges):
        attractive_source_rgba = [0.9, 0.8, 0.7, 0.8]
        attractive_field_rgba = [0.9, 0.8, 0.7, 0.2]
        repulsive_source_rgba = [0.1, 0.2, 0.3, 0.8]
        repulsive_field_rgba = [0.1, 0.2, 0.3, 0.2]

        for i in range(len(attractive_forces)):
            self.viewer.add_marker(
                label='',
                type=const.GEOM_SPHERE,
                size=np.array([0.01, 0.01, 0.01]),
                pos=np.array(attractive_forces[i]),
                rgba=np.array(attractive_source_rgba),
                mat=np.eye(3).flatten()
            )
        for i in range(len(attractive_forces)):
            sz = attractive_force_ranges[i]
            self.viewer.add_marker(
                label='',
                type=const.GEOM_SPHERE,
                size=np.array([sz, sz, sz]),
                pos=np.array(attractive_forces[i]),
                rgba=np.array(attractive_field_rgba),
                mat=np.eye(3).flatten()
            )

        for i in range(len(repulsive_forces)):
            self.viewer.add_marker(
                label='',
                type=const.GEOM_SPHERE,
                size=np.array([0.01, 0.01, 0.01]),
                pos=np.array(repulsive_forces[i]),
                rgba=np.array(repulsive_source_rgba),
                mat=np.eye(3).flatten()
            )
        for i in range(len(repulsive_forces)):
            sz = repulsive_force_ranges[i]
            self.viewer.add_marker(
                label='',
                type=const.GEOM_SPHERE,
                size=np.array([sz, sz, sz]),
                pos=np.array(repulsive_forces[i]),
                rgba=np.array(repulsive_field_rgba),
                mat=np.eye(3).flatten()
            )


class SawyerReachingEnvGoal(SawyerReachingEnv):

    """
    This env is the multi-task version of reaching with different position of the goal
    """

    def __init__(
        self,
        is_eval_env=False,
        reward_scaling=1.0,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        use_force=False,
        **kwargs
    ):
        self.goal_range = Box(
            low=np.array([0.75, -0.3, 0.3]), high=np.array([0.9, 0.3, 0.7])
        )

        if goal_site_name is None:
            goal_site_name = "goal_reach_site"

        if is_eval_env:
            self.task_list = self.generate_list_of_tasks(
                num_tasks=10, is_eval_env=is_eval_env
            )
        else:
            self.task_list = self.generate_list_of_tasks(
                num_tasks=30, is_eval_env=is_eval_env
            )

        super(SawyerReachingEnvGoal, self).__init__(
            xml_path=xml_path,
            goal_site_name=goal_site_name,
            action_mode=action_mode,
            reward_scaling=reward_scaling,
            use_force=use_force
        )

    def reset_task(self, task=None):
        if task is None:
            task_idx = np.random.randint(len(self.task_list))
        else:
            task_idx = task
        if np.isscalar(task_idx):
            self.set_task(self.task_list[task_idx])
        else:
            self.set_task(task_idx)

    def reset(self):
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def generate_list_of_tasks(self, num_tasks, is_eval_env):
        """To be called externally to obtain samples from the task distribution"""
        if is_eval_env:
            np.random.seed(100)  # pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        possible_goals = [self.goal_range.sample() for _ in range(num_tasks)]
        return possible_goals

    def set_task(self, goal):
        """To be called externally to set the task for this environment"""
        self.model.site_pos[self.site_id_goal] = goal.copy()

    def get_task(self):
        return self.model.site_pos[self.site_id_goal]


class SawyerReachingEnvGoalOOD(SawyerReachingEnv):
    def __init__(
        self,
        is_eval_env=False,
        reward_scaling=1.0,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        distance=0.0,
        **kwargs
    ):

        self.distance = distance
        if distance > 0.6:
            raise Exception("Cannot handle distance > 0.6")
        second_bound = 0.3 - distance

        if not is_eval_env:
            self.goal_range = Box(
                low=np.array([0.75, -0.3, 0.3]), high=np.array([0.9, second_bound, 0.7])
            )
        else:
            self.goal_range = Box(
                low=np.array([0.75, -second_bound, 0.3]), high=np.array([0.9, 0.3, 0.7])
            )

        if goal_site_name is None:
            goal_site_name = "goal_reach_site"

        self.task_list = self.generate_list_of_tasks(
            num_tasks=30, is_eval_env=is_eval_env
        )

        super(SawyerReachingEnvGoalOOD, self).__init__(
            xml_path=xml_path,
            goal_site_name=goal_site_name,
            action_mode=action_mode,
            reward_scaling=reward_scaling,
        )

    def reset_task(self, task=None):
        if task is None:
            task_idx = np.random.randint(len(self.task_list))
        else:
            task_idx = task
        self.set_task(self.task_list[task_idx])

    def reset(self):
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def generate_list_of_tasks(self, num_tasks, is_eval_env):
        """To be called externally to obtain samples from the task distribution"""
        if is_eval_env:
            np.random.seed(100)  # pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        possible_goals = [self.goal_range.sample() for _ in range(num_tasks)]
        np.random.seed()
        return possible_goals

    def set_task(self, goal):
        """To be called externally to set the task for this environment"""
        self.model.site_pos[self.site_id_goal] = goal.copy()

    def get_task(self):
        return self.model.site_pos[self.site_id_goal]


class SawyerReachingEnvGoalComplexTwoModes(SawyerReachingEnv):
    def __init__(
        self,
        is_eval_env=False,
        reward_scaling=1.0,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        distance=0.0,
        **kwargs
    ):

        self.distance = distance
        if distance > 0.6:
            raise Exception("Cannot handle distance > 0.6")
        second_bound = np.abs(distance - 0.3)

        self.goal_range_1 = Box(
            low=np.array([0.75, -0.3, 0.3]), high=np.array([0.9, -second_bound, 0.7])
        )
        self.goal_range_2 = Box(
            low=np.array([0.75, second_bound, 0.3]), high=np.array([0.9, 0.3, 0.7])
        )

        if goal_site_name is None:
            goal_site_name = "goal_reach_site"

        self.task_list = self.generate_list_of_tasks(num_tasks=100)

        self.action_perturb = 0.0

        super().__init__(
            xml_path=xml_path,
            goal_site_name=goal_site_name,
            action_mode=action_mode,
            reward_scaling=reward_scaling,
        )

    def generate_list_of_tasks(self, num_tasks):
        """To be called externally to obtain samples from the task distribution"""

        goal_modes = [self.goal_range_1, self.goal_range_1]
        action_modes = [0.05, -0.05]

        possible_goals = []
        for _ in range(num_tasks):
            mode = np.random.choice([0, 1])
            possible_goals.append(
                (goal_modes[mode].sample(), np.random.normal(action_modes[mode], 0.05))
            )

        return possible_goals

    def set_task(self, goal):
        """To be called externally to set the task for this environment"""
        self.model.site_pos[self.site_id_goal] = goal[0].copy()
        self.action_perturb = goal[1]

    def get_task(self):
        return (self.model.site_pos[self.site_id_goal], self.action_perturb)

    def step(self, action):
        """ apply the 7DoF action provided by the policy """
        return super().step(action + self.action_perturb)
