# Author: Mattia Silvestri

"""
    RL policies.
"""

import random
import numpy as np

########################################################################################################################


class CategoricalPolicy:
    """
    Abstract class for a categorical policy.
    """

    def __init__(self, actions_space):
        """
        :param actions_space: int; number of actions.
        """

        self._num_actions = actions_space

    @property
    def num_actions(self):
        """
        Getter for the actions space.
        :return: int; the actions space.
        """

        return self._num_actions

    def select_action(self, *args, **kwargs):
        """
        Select the action according to the strategy.
        """

        raise NotImplementedError()

########################################################################################################################


class CategoricalRandomPolicy(CategoricalPolicy):
    """
    Actions are randomly chosen.
    """

    def __init__(self, actions_space):
        """
        :param actions_space: int; number of actions.
        """

        super(CategoricalRandomPolicy, self).__init__(actions_space)

    def select_action(self):
        """
        Select the action according to the strategy.
        """

        action = random.randint(0, self.num_actions-1)
        return action

########################################################################################################################


class CategoricalGreedyPolicy(CategoricalPolicy):
    """
    Always chose the best action.
    """

    def select_action(self, q_values):
        """
        Select the best action accoring to the Q-values.
        :param q_values: numpy.array of shape (n_actions, ); the Q-values.
        :return:
        """
        action = np.argmax(q_values)
        return action

########################################################################################################################


class CategoricalStochasticPolicy(CategoricalPolicy):
    """
    Sample action according to the probabilities values given as input.
    """

    def select_action(self, probs):
        """
        Select the action according to the probability values given as input.
        :param probs: numpy.array of shape (n_actions,); the probability to choose each action.
        :return:
        """
        actions = np.arange(0, self.num_actions)
        action = np.random.choice(actions, size=1, p=probs)

        return action[0]

########################################################################################################################


class EpsilonGreedyPolicy(CategoricalPolicy):
    """
    Epsilon greedy policy with epsilon linear annealing
    """
    def __init__(self, actions_space, epsilon_start, epsilon_end, nb_steps):
        """

        :param actions_space: number of actions; as integer
        :param epsilon_start: initial value for epsilon; as float
        :param epsilon_end: final value for epsilon; as float
        :param nb_steps: number of steps for the annealing of epsilon; as integer
        """
        super(EpsilonGreedyPolicy, self).__init__(actions_space)
        self._count = 1

        # Create the parameters for the linear annealing: f(x) = ax + b
        self._angolar_coeff_anneal = -float(epsilon_start - epsilon_end) / float(nb_steps)
        self._bias_coeff_anneal = float(epsilon_start)
        self._epsilon_end = float(epsilon_end)
        self._epsilon = epsilon_start

    @property
    def epsilon(self):
        """
        Getter for the current epsilon value.
        :return: float; the current epsilon value.
        """

        return self._epsilon

    def select_action(self, q_values):
        """
        With probability epsilon choose a random action; with probability (1-epsilon) choose the best action.
        :param q_values: numpy.array; the Q-values.
        :return:
        """
        if np.random.uniform() <= self._epsilon:
            action = random.randint(0, self.num_actions-1)
        else:
            action = np.argmax(q_values)

        self._epsilon = max(self._epsilon_end,
                            self._angolar_coeff_anneal * float(self._count) + self._bias_coeff_anneal)

        self._count += 1

        return action

