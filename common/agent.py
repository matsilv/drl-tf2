# Author: Mattia Silvestri

"""
    RL agents.
"""

import numpy as np

from common.utility import calc_qvals
from common.memory import ReplayExperienceBuffer

########################################################################################################################


class DRLAgent:
    """
    Abstract class for Deep Reinforcement Learning agent.
    """

    def __init__(self, env, policy, model):
        """

        :param env: gym.Environment; the agent interacts with this environment.
        :param policy: policy.Policy; policy defined as a probability distribution of actions over states.
        :param model: model.DRLModel; DRL model.
        """

        self._env = env
        self._policy = policy
        self._model = model

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; number of interactions with the environment for training.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where to save/load model weights.
        :return:
        """

        raise NotImplementedError()

    def test(self, loadpath, render):
        """
        Test the model.
        :param loadpath: string; model weights loadpath.
        :param render: bool; True if you want to visualize the environment, False otherwise.
        :return:
        """

        # Load model weights
        self._model.load_weights(loadpath)

        while True:

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()
            score = 0

            # Perform an episode
            while not game_over:

                if render:
                    self._env.render()

                # Sample an action from policy
                probs = self._model.act(np.expand_dims(s_t, axis=0))
                probs = np.squeeze(probs.numpy())
                a_t = self._policy.select_action(probs)

                # Perform a step
                s_tp1, r_t, game_over, _ = self._env.step(a_t)
                s_t = s_tp1

                score += 1

            print('Score: {}'.format(score))

########################################################################################################################


class OnPolicyAgent(DRLAgent):
    """
    DRL agent which requires on-policy samples.
    """

    def __init__(self, env, policy, model):
        """

        :param env: environment on which to train the agent; as Gym environment
        :param policy: policy defined as a probability distribution of actions over states; as policy.Policy
        :param model: DRL model; as models.DRLModel
        """

        super(OnPolicyAgent, self).__init__(env, policy, model)

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; training steps in the environment.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where model weights will be saved.
        :return:
        """

        # Training steps
        steps = 0

        # Sampled trajectory variables
        rewards = []
        actions = []
        states = []
        q_vals = []

        while steps < num_steps:

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()
            score = 0

            current_states = []
            current_actions = []
            current_q_vals = []
            current_rewards = []

            # Perform an episode
            while not game_over:

                if render:
                    self._env.render()

                # Sample an action from policy
                probs = self._model.act(np.expand_dims(s_t, axis=0))
                probs = np.squeeze(probs.numpy())
                a_t = self._policy.select_action(probs)
                action = np.zeros(self._env.action_space.n)
                action[a_t] = 1
                current_actions.append(action)

                # Sample current state, next state and reward
                current_states.append(s_t)
                s_tp1, r_t, game_over, _ = self._env.step(a_t)
                current_rewards.append(r_t)
                s_t = s_tp1

                # Increase the score and the steps counter
                score += r_t
                steps += 1

            # Compute the Q-values
            current_q_vals = calc_qvals(current_rewards, gamma=gamma)

            states = states + current_states
            actions = actions + current_actions
            q_vals = q_vals + current_q_vals

            if len(states) > batch_size:
                states = np.asarray(states)[:batch_size]
                q_vals = np.asarray(q_vals)[:batch_size]
                actions = np.asarray(actions)[:batch_size]

                # Perform a gradient descent step
                loss = self._model.train_step(states, q_vals, actions)

                print('Frame: {}/{} | Score: {} | Loss policy: {}'.
                      format(steps, num_steps, score, loss))

                # Clear sample trajectory variables
                states = []
                rewards = []
                actions = []
                q_vals = []

        # Save model weights
        self.model.save_weights(filename, save_format='tf')


class OffPolicyAgent(DRLAgent):
    """
    DRL agent which can be trained using off-policy samples.
    """

    def __init__(self, env, policy, model):
        """

        :param env: environment on which to train the agent; as Gym environment
        :param policy: policy defined as a probability distribution of actions over states; as policy.Policy
        :param model: DRL model; as models.DRLModel
        """

        super(OffPolicyAgent, self).__init__(env, policy, model)

        self.buffer = ReplayExperienceBuffer(maxlen=50000)

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: training steps in the environment; as int
        :param render: True if you want to render the environment while training; as boolean
        :param gamma: discount factor; as double
        :param batch_size: batch size; as int
        :param filename: file path where model's weights will be saved; as string
        :return:
        """

        # Training steps
        steps = 0

        loss = 0.0

        while steps < num_steps:

            # Initialize the environment
            game_over = False
            s_t = self.env.reset()
            score = 0

            # Perform an episode
            while not game_over:

                if render:
                    self.env.render()

                # Sample an action from policy
                probs = self.model.act(s_t.reshape(1, *s_t.shape))
                probs = probs.numpy().reshape(-1)
                a_t = self.policy.select_action(probs)
                action = np.zeros(self.env.action_space.n)
                action[a_t] = 1

                # Sample current state, next state and reward
                s_tp1, r_t, game_over, _ = self.env.step(a_t)
                s_tp1 = np.array(s_tp1)

                # Insert sample in the buffer
                self.buffer.insert((np.array(s_t), a_t, r_t, s_tp1, game_over))

                # Sample a mini-batch from experience and perform an update
                if len(self.buffer) > batch_size:
                    batch = self.buffer.get_random_batch(batch_size)
                    loss = self.model.gradient_step(batch, gamma)

                s_t = s_tp1

                score += r_t
                steps += 1
            print('Frame: {}/{} | Score: {} | Loss: {} | Epsilon: {}'.format(steps, num_steps, score, loss,
                                                                             self.policy.epsilon))

        # Save model weights
        self.model.save_weights(filename, save_format='tf')