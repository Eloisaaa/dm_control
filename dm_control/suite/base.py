# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Base class for tasks in the Control Suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control.utils import inverse_kinematics
from IPython import embed
from dm_control import mujoco
from dm_control.rl import control

import numpy as np


class Task(control.Task):
  """Base class for tasks in the Control Suite.

  Actions are mapped directly to the states of MuJoCo actuators: each element of
  the action array is used to set the control input for a single actuator. The
  ordering of the actuators is the same as in the corresponding MJCF XML file.

  Attributes:
    random: A `numpy.random.RandomState` instance. This should be used to
      generate all random variables associated with the task, such as random
      starting states, observation noise* etc.

  *If sensor noise is enabled in the MuJoCo model then this will be generated
  using MuJoCo's internal RNG, which has its own independent state.
  """

  def __init__(self, random=None):
    """Initializes a new continuous control task.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an integer
        seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    if not isinstance(random, np.random.RandomState):
      random = np.random.RandomState(random)
    self._random = random
    self._visualize_reward = False

  @property
  def random(self):
    """Task-specific `numpy.random.RandomState` instance."""
    return self._random

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    return mujoco.action_spec(physics)

  def initialize_episode(self, physics):
    """Resets geom colors to their defaults after starting a new episode.

    Subclasses of `base.Task` must delegate to this method after performing
    their own initialization.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    self.after_step(physics)

  def before_step(self, action, physics):
    """Sets the control signal for the actuators to values in `action`."""
    # Support legacy internal code.
    action = getattr(action, "continuous_actions", action)
    physics.set_control(action)

  def after_step(self, physics):
    """Modifies colors according to the reward."""
    if self._visualize_reward:
      reward = np.clip(self.get_reward(physics), 0.0, 1.0)
      _set_reward_colors(physics, reward)
  def set_site_to_xpos(self, physics, random_state, site, fence, target_pos,
                       target_quat=None, max_ik_attempts=10):
    """Moves the arm so that a site occurs at the specified location.
    This function runs the inverse kinematics solver to find a configuration
    arm joints for which the pinch site occurs at the specified location in
    Cartesian coordinates.
    Args:
      physics: A `mujoco.Physics` instance.
      random_state: An `np.random.RandomState` instance.
      site: Either a `mjcf.Element` or a string specifying the full name
        of the site whose position is being set.
      target_pos: The desired Cartesian location of the site.
      target_quat: (optional) The desired orientation of the site, expressed
        as a quaternion. If `None`, the default orientation is to point
        vertically downwards.
      max_ik_attempts: (optional) Maximum number of attempts to make at finding
        a solution satisfying `target_pos` and `target_quat`. The joint
        positions will be randomized after each unsuccessful attempt.
    Returns:
      A boolean indicating whether the desired configuration is obtained.
    Raises:
      ValueError: If site is neither a string nor an `mjcf.Element`.
    """
    if isinstance(site, mjcf.Element):
      site_name = site.full_identifier
    elif isinstance(site, str):
      site_name = site
    else:
      raise ValueError('site should either be a string or mjcf.Element: got {}'
                       .format(site))
    if target_quat is None:
      target_quat = DOWN_QUATERNION
    lower = [fence['x'][0],fence['y'][0],fence['z'][0]]
    upper = [fence['x'][-1],fence['y'][-1],fence['z'][-1]]
    arm_joint_names = ['jaco_joint_1','jaco_joint_2','jaco_joint_3','jaco_joint_4','jaco_joint_5','jaco_joint_6','jaco_joint_7']

    for _ in range(max_ik_attempts):
      result = inverse_kinematics.qpos_from_site_pose(
          physics=physics,
          site_name=site_name,
          target_pos=target_pos,
          target_quat=target_quat,
          joint_names=arm_joint_names,
          rot_weight=2,
          inplace=True)
      success = result.success

      # Canonicalise the angle to [0, 2*pi]
      if success:
        for arm_joint, low, high in zip(arm_joint_names, lower, upper):
          while physics.named.data.geom_xpos[arm_joint] >= high:
            physics.named.data.qpos[arm_joint] -= 2*np.pi
          while physics.named.data.geom_xpos[arm_joint] < low:
            physics.named.data.qpos[arm_joint] += 2*np.pi
            if physics.named.data.geom_xpos[arm_joint] > high:
              success = False
              break

      # If succeeded or only one attempt, break and do not randomize joints.
      if success or max_ik_attempts <= 1:
        break
      else:
        #self.randomize_arm_joints(physics, random_state)

    return success

  @property
  def visualize_reward(self):
    return self._visualize_reward

  @visualize_reward.setter
  def visualize_reward(self, value):
    if not isinstance(value, bool):
      raise ValueError("Expected a boolean, got {}.".format(type(value)))
    self._visualize_reward = value


_MATERIALS = ["self", "effector", "target"]
_DEFAULT = [name + "_default" for name in _MATERIALS]
_HIGHLIGHT = [name + "_highlight" for name in _MATERIALS]


def _set_reward_colors(physics, reward):
  """Sets the highlight, effector and target colors according to the reward."""
  assert 0.0 <= reward <= 1.0
  colors = physics.named.model.mat_rgba
  default = colors[_DEFAULT]
  highlight = colors[_HIGHLIGHT]
  blend_coef = reward ** 4  # Better color distinction near high rewards.
  colors[_MATERIALS] = blend_coef * highlight + (1.0 - blend_coef) * default
