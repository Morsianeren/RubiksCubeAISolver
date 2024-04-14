# %% Imports
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics

from RubiksCubeSimulation.RubiksCubeEnvironment import RubiksCubePyEnvironment

# %% Create the environment

env = tf_py_environment.TFPyEnvironment(RubiksCubePyEnvironment())

#tf.random.uniform(action_spec.minimum, action_spec.maximum, dtype=tf.int32)

# %% Define the QNetwork

fc_layer_params = (100,)

q_net = q_network.QNetwork(
    env.observation_spec(),
    action_spec=env.action_spec(),
    fc_layer_params=fc_layer_params
)

# %% Define the DQN Agent

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

# %% Define the replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=1000 # NOTE: there  are 43 quintillion possible combinations of a 3x3x3 Rubik's cube
)

# %% Define the data collection driver

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=1 # Collect data until the buffer is full
)

# %% Collect initial data

collect_driver.run()

# %% Define the dataset

# Note we only do one step per episode

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)

# %% Define the training loop

def train_agent(n_iterations):
    for _ in range(n_iterations):
        collect_driver.run()
        experience, _ = next(iter(dataset))
        train_loss = agent.train(experience).loss
        print(f"Loss at iteration {_}: {train_loss}")

# %% Train the agent

train_agent(100)

# %% Save the agent

tf_policy = agent.policy
#saved_policy = tf_policy.get_initial_policy()
#saved_policy
