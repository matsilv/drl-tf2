"""
    Main script.
"""

import gym
import argparse
import tensorflow as tf

from common.models import A2C, PolicyGradient, DeepQLearning, DDPG
from common.atari_wrapper import make_env, NormalizedEnv
from common.agent import OffPolicyAgent, OnPolicyAgent
from common.policy import StochasticPolicy, EpsilonGreedyPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--alg", type=str, choices=["a2c", "pg", "dqn", "ddpg"],  help="Choose the algorithm")
parser.add_argument("--env-name", type=str, help="Gym registered environment")
parser.add_argument("--render", action="store_true", default=False, help="Human rendering of the environment")
parser.add_argument("--num-steps", type=int, default=100000, help="Number of timesteps in the environment")
parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--atari", action="store_true", default=False, help="True if your are using an Atari environment")
parser.add_argument("--gpu", action="store_true", default=False, help="Allow TensorFlow GPU computation")
parser.add_argument("--test", action="store_true", default=False, help="Test on an existing model")
parser.add_argument("--filepath", type=str, help="File path where model weights will be saved/loaded")

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
    if isinstance(env.action_space, gym.spaces.Box):
        env = NormalizedEnv(env)
else:
    env = make_env(env_name)

render = args.render
num_steps = int(args.num_steps)
gamma = float(args.gamma)
batch_size = int(args.batch_size)
test = args.test
filepath = args.filepath
input_shape=env.observation_space.shape

if args.alg == "a2c":
    model = A2C(input_shape=input_shape, output_dim=env.action_space.n, hidden_units=[32, 32], atari=args.atari)
    policy = StochasticPolicy(env.action_space.n)
    agent = OnPolicyAgent(env, policy, model)
elif args.alg == "pg":
    model = PolicyGradient(input_shape=input_shape, output_dim=env.action_space.n, hidden_units=[256, 128], atari=args.atari)
    policy = StochasticPolicy(env.action_space.n)
    agent = OnPolicyAgent(env, policy, model)
elif args.alg == "dqn":
    model = DeepQLearning(input_shape=input_shape, output_dim=env.action_space.n, hidden_units=[32, 32], atari=args.atari)
    policy = EpsilonGreedyPolicy(env.action_space.n, epsilon_start=1.0, epsilon_end=0.1, nb_steps=10000)
    agent = OffPolicyAgent(env, policy, model)
elif args.alg == "ddpg":
    agent = DDPG(num_states=env.observation_space.shape[0], num_actions=env.action_space.shape[0],
                 actor_hidden_units=[32, 32], critic_hidden_units=[256, 256], memory_size=50000, tau=0.99)
    agent.train(env, num_steps, render, gamma, batch_size=128)
    exit(0)
else:
    raise Exception("Algorithm not recognized")

if not test:
    agent.train(num_steps, render, gamma, batch_size, filepath)
else:
    agent.test(filepath, render)

'''elif args.alg == "ddpg":
    agent = DDPG(num_states=env.observation_space.shape[0], num_actions=env.action_space.shape[0], actor_hidden_units=[32, 32],
                 critic_hidden_units=[256, 256])
    agent.train(env, num_steps, render, gamma, batch_size=128)'''