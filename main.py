"""
    Main script.
"""

import gym
import argparse
import tensorflow as tf
from common.models import PolicyGradient
from common.atari_wrapper import NormalizedEnv
from common.agent import OnPolicyAgent
from common.policy import CategoricalStochasticPolicy

########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, choices=["pg"],  help="Choose the algorithm")
    parser.add_argument("--env-name", type=str, help="Gym registered environment")
    parser.add_argument("--render", action="store_true", default=False, help="Human rendering of the environment")
    parser.add_argument("--num-steps", type=int, default=100000, help="Number of timesteps in the environment")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
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
    else:
        env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Box):
        env = NormalizedEnv(env)

    render = args.render
    num_steps = int(args.num_steps)
    gamma = float(args.gamma)
    batch_size = int(args.batch_size)
    test = args.test
    filepath = args.filepath

    if args.alg == "pg":
        model = PolicyGradient(input_shape=env.observation_space.shape,
                               output_dim=env.action_space.n)
        policy = CategoricalStochasticPolicy(env.action_space.n)
        agent = OnPolicyAgent(env, policy, model)
    else:
        raise Exception("Algorithm not recognized")

    if not test:
        agent.train(num_steps, render, gamma, batch_size, filepath)
    else:
        agent.test(filepath, render)