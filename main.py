"""
    Main script.
"""

import gym
import argparse
import tensorflow as tf

from common.models import A2CNetwork, DDPG
from common.atari_wrapper import make_env, NormalizedEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str, choices=["a2c", "ddpg"],  help="Choose the algorithm")
parser.add_argument("--env-name", type=str, help="Gym registered environment")
parser.add_argument("--render", action="store_true", default=False, help="Human rendering of the environment")
parser.add_argument("--num-steps", type=int, default=100000, help="Number of timesteps in the environment")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--atari", action="store_true", default=False, help="True if your are using an Atari environment")
parser.add_argument("--gpu", action="store_true", default=False, help="Allow TensorFlow GPU computation")

args = parser.parse_args()

# GPU setup
if args.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Parsing arguments
env_name = args.env_name

if env_name is None:
    raise Exception("Missing environment name")
    exit(1)

if not args.atari:
    env = gym.make(env_name)
    env = NormalizedEnv(env)
else:
    env = make_env(env_name)

render = args.render
num_steps = int(args.num_steps)
gamma = float(args.gamma)

if args.alg == "a2c":
    agent = A2CNetwork(output_dim=env.action_space.n, hidden_units=[32, 32], atari=args.atari)
    agent.train(env, num_steps, render, gamma)
elif args.alg == "ddpg":
    agent = DDPG(num_states=env.observation_space.shape[0], num_actions=env.action_space.shape[0], actor_hidden_units=[32, 32],
                 critic_hidden_units=[256, 256])
    agent.train(env, num_steps, render, gamma, batch_size=128)
else:
    raise Exception("Algorithm not recognized")