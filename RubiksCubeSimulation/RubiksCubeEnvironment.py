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
from .Cube.RubiksCube import RubiksCube
#from tensorflow.agents.specs import array_spec

# %% Main class

class RubiksCubePyEnvironment(PyEnvironment):
    def __init__(self, maximum_steps=1, scrambles=1):
        # The algorithm need to decide 3 things:
        # - which axis to rotate around [x, y, z]
        # - which row to rotate [0, 1, 2]
        # - direction of rotation [-1, 1]
        # That gives us a total of 18 possible actions
        self._action_spec = BoundedArraySpec(
            shape=(),
            minimum=0,
            maximum=17,
            dtype=np.int32,
            name='action'
        )

        # The observation is the current state of the cube
        # The state is a 6x3x3=54 array of integers
        self._observation_spec = BoundedArraySpec(
            shape=(54,),
            dtype=np.int32,
            minimum=0,
            name='observation'
        )
        
        self._state = RubiksCube()
        self._episode_ended = False
        self.maximum_steps = maximum_steps
        self._step_counter = 0
        self.scrambles = scrambles # The number of rotations on the cube to do before starting the episode
        self._completed_faces = set() # The faces that have been completed for the face completion reward function

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        #self._state.reset()
        self._state.scramble(self.scrambles) # Scramble the cube (can be done without resetting the cube)
        self._episode_ended = False
        self._step_counter = 0

        # Reset reward function
        _, completions = face_completion_rf(self._state, set())
        self._completed_faces = completions # Don't give rewards for faces that have already been completed

        observation = self._state.array(binary=True).flatten()
        return ts.restart(observation)

    def _step(self, action):
        self._step_counter += 1
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # Update the state based on the action
        #axis = action // 6
        #row = (action % 6) // 3
        #k = action % 3 + 1
        #self._state.rotate_side(axis, row, k)
        # TODO: Implement the action
        observation = self._state.array(binary=True).flatten()
        reward = self._reward_function()

        # Make sure episodes don't go on forever.
        if self._step_counter >= self.maximum_steps:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(
                observation=observation,
                reward=reward
            )
        else:
            return ts.transition(
                observation=observation,
                reward=reward,
                discount=1.0
            )
        
    def render(self):
        self._state.plot()
        
    def _reward_function(self):
        reward, new_completions = face_completion_rf(self._state, self._completed_faces)
        self._completed_faces.update(new_completions)
        return reward
    

def face_completion_rf(cube: RubiksCube, completed_faces: set):
    """Reward function that gives a reward of 1 if a face is completed, 0 otherwise.
    Note that this function can be exploited by rotating the same face back and forth, 
    thereby repeatedly getting a positive reward"""
    # The array() function return an array with a list of colors for each face.
    # Example:
    # array([['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
    #   ...
    #   ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']], dtype='<U1')
    colors = cube.array()

    reward = 0
    new_completions = set()

    # Check if all colors of a face are the same
    for i in range(6):
        face_colors = colors[i] # Get the colors of the face
        if len(set(face_colors)) == 1: # If len == 1, then all the values are the same
            if i not in completed_faces: # Check if the face has already been completed
                reward += 1
                new_completions.add(i)
                break
        
    return (reward, new_completions)

# %% Test code
# env = RubiksCubePyEnvironment()
# env.reset()
# env.render()
# %%
