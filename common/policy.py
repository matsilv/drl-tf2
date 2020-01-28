# Author: Mattia Silvestri

import random
import numpy as np


# abstract class for policies
class Policy:

    def __init__(self, actions_space):
        self.num_actions = actions_space

    def select_action(self, **kwargs):
        raise NotImplementedError()


# random policy
class RandomPolicy(Policy):
    def select_action(self, **kwargs):
        action = random.randint(0, self.num_actions-1)
        return action


# greedy policy
class GreedyPolicy(Policy):
    def select_action(self, q_values, **kwargs):
        action = np.argmax(q_values)
        return action


class StochasticPolicy(Policy):
    def select_action(self, probs, **kwargs):
        actions = np.arange(0, self.num_actions)
        action = np.random.choice(actions, size=1, p=probs)

        return action[0]


# epsilon greedy policy with epsilon linear annealing
class EpsilonGreedyPolicy(Policy):
    def __init__(self, actions_space, epsilon_start, epsilon_end, nb_steps):
        super(EpsilonGreedyPolicy, self).__init__(actions_space)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.nb_steps = nb_steps
        self.count = 1

    def select_action(self, q_values, **kwargs):
        if np.random.uniform() <= self.epsilon:
            action = random.randint(0, self.num_actions-1)
        else:
            action = np.argmax(q_values)

        # Linear annealed: f(x) = ax + b.
        a = -float(self.epsilon_start - self.epsilon_end) / float(self.nb_steps)
        b = float(self.epsilon_start)
        self.epsilon = max(self.epsilon_end, a * float(self.count) + b)

        #print("Epsilon: {}".format(self.epsilon))

        self.count += 1

        return action


"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

