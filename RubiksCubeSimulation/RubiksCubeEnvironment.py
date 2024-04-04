# %% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.specs import BoundedArraySpec
from Cube.RubiksCube import RubiksCube
#from tensorflow.agents.specs import array_spec

# %% Main class

class RubiksCubePyEnvironment(PyEnvironment):
    def __init__(self):
        # In our action spec, we need to know which axis [x, y, z], row [0, 1, 2] and rotations [-1, 1]
        # Define the ranges for each action component
        # Define action specification for axis (x, y, z)
        axis_spec = BoundedArraySpec(
            name='axis',
            minimum=0,
            maximum=2,
            dtype=tf.int32
        )
        # Define action specification for row (0, 1, 2)
        row_spec = BoundedArraySpec(
            name='row',
            minimum=0,
            maximum=2,
            dtype=tf.int32
        )
        rotation_spec = BoundedArraySpec(
            name='rotation',
            minimum=0, # -1
            maximum=1, # 1
            dtype=tf.int32
        )

        # Create the action spec using a nested dictionary
        self._action_spec = {
            'axis': axis_spec,
            'row': row_spec,
            'rotation': rotation_spec
        }
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
        

# %% Test code
env = RubiksCubePyEnvironment()
