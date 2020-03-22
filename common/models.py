"""
    Define policy and critic model and custom loss function with TensorFlow 2.
"""

import tensorflow as tf
import numpy as np

import os
cwd = os.getcwd()
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '{}/../'.format(cwd))

from common.policy import StochasticPolicy, GreedyPolicy, OUNoise
from common.memory import ReplayExperienceBuffer


########################################################################################################################


def calc_qvals(rewards, gamma):
    """
    Compute expected Q-values.
    :param rewards: list with episode rewards
    :param gamma: discount factor as double
    :return: expected Q-values as list
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)

    return list(reversed(res))

########################################################################################################################


def __compute_pg_loss__(policies, next_estimated_return, actions):
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=policies, labels=actions)
    policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, next_estimated_return))

    return policy_loss

########################################################################################################################


class PolicyGradient(tf.keras.layers.Layer):
    """
        Definition of Policy Gradient model and custom training loop.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False):
        """
        :param output_dim: output dimension as integer
        """

        super(PolicyGradient, self).__init__()
        self.output_dim = output_dim

        # Define common body
        self.model = []
        if not atari:
            for hidden in hidden_units:
                self.model.append(tf.keras.layers.Dense(units=hidden, activation='relu'))
        else:
            self.model.append(tf.keras.layers.Conv2D(filters=32,
                                                           kernel_size=[8, 8],
                                                           strides=[4, 4],
                                                           activation='relu'))
            self.model.append(tf.keras.layers.Conv2D(filters=64,
                                                           kernel_size=[4, 4],
                                                           strides=[2, 2],
                                                           activation='relu'))
            self.model.append(tf.keras.layers.Conv2D(filters=64,
                                                           kernel_size=[3, 3],
                                                           strides=[1, 1],
                                                           activation='relu'))

            self.model.append(tf.keras.layers.Flatten())
            self.model.append(tf.keras.layers.Dense(512))

        # Define actor and critic
        self.model.append(tf.keras.layers.Dense(output_dim))

        # Define optimizers
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, x):
        """
        Implement call method of tf.keras Layer.
        :param x: inputs as tf.Tensor
        :return: policy, value state and probabilities as tf.Tensor
        """

        for l in self.model:
            x = l(x)

        probs = tf.nn.softmax(x)

        return x, probs

    def train(self, env, num_steps, render, gamma):
        """
        Training loop.
        :param env: gym environment
        :param num_steps: training steps in the environment as int
        :param render: True if you want to render the environment while training
        :param gamma: discount factor as double
        :return:
        """

        frames = 0

        rewards = []
        actions = []
        states = []

        # Create policy
        policy = StochasticPolicy(env.action_space.n)

        count = 0

        while frames < num_steps:
            game_over = False
            s_t = env.reset()
            score = 0

            while not game_over:

                if render:
                    env.render()

                logits, probs = self.call(s_t.reshape(1, *s_t.shape))
                probs = probs.numpy().reshape(-1)
                a_t = policy.select_action(probs)
                action = np.zeros(env.action_space.n)
                action[a_t] = 1
                actions.append(action)
                states.append(s_t)
                s_tp1, r_t, game_over, _ = env.step(a_t)
                rewards.append(r_t)
                s_tp1 = np.array(s_tp1)
                s_t = s_tp1

                score += r_t
                frames += 1

            print('Epochs: {} | Reward: {}'.format(frames, score))

            q_vals = calc_qvals(rewards, gamma=gamma)
            states = np.asarray(states)
            q_vals = np.asarray(q_vals)
            actions = np.asarray(actions)

            with tf.GradientTape(persistent=True) as tape:
                policies, probs = self.call(states)
                policy_loss = __compute_pg_loss__(policies, q_vals, actions)

            dloss_policy = tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(dloss_policy, self.trainable_variables))

            print('Frame: {}/{} | Score: {} | Loss policy: {}'.
                  format(frames, num_steps, score, policy_loss))

            states = []
            rewards = []
            actions = []

            count += 1




########################################################################################################################

def __compute_a2c_loss__(policies, probs, values, next_estimated_return, actions):
    advantage = next_estimated_return - tf.stop_gradient(values)
    neg_log_prob = advantage * tf.nn.softmax_cross_entropy_with_logits(logits=policies, labels=actions)
    policy_loss = tf.reduce_mean(neg_log_prob * next_estimated_return)
    value_loss = tf.reduce_mean((values - next_estimated_return) ** 2)
    entropy_loss = 0.01 * tf.reduce_mean(tf.reduce_sum(probs * tf.nn.log_softmax(policies), axis=1))

    return policy_loss, value_loss, entropy_loss

########################################################################################################################


class A2CNetwork(tf.keras.layers.Layer):
    """
        Definition of A2C model and custom training loop.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False):
        """
        :param output_dim: output dimension as integer
        """

        super(A2CNetwork, self).__init__()
        self.output_dim = output_dim

        # Define common body
        self.common_body = []
        if not atari:
            for hidden in hidden_units:
                self.common_body.append(tf.keras.layers.Dense(units=hidden, activation='relu'))
        else:
            self.common_body.append(tf.keras.layers.Conv2D(filters=32,
                                                           kernel_size=[8, 8],
                                                           strides=[4, 4],
                                                           activation='relu'))
            self.common_body.append(tf.keras.layers.Conv2D(filters=64,
                                                           kernel_size=[4, 4],
                                                           strides=[2, 2],
                                                           activation='relu'))
            self.common_body.append(tf.keras.layers.Conv2D(filters=64,
                                                           kernel_size=[3, 3],
                                                           strides=[1, 1],
                                                           activation='relu'))

            self.common_body.append(tf.keras.layers.Flatten())
            self.common_body.append(tf.keras.layers.Dense(512))

        # Define actor and critic
        self.actor = tf.keras.layers.Dense(output_dim)
        self.critic = tf.keras.layers.Dense(1)

        # Define optimizers
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, x):
        """
        Implement call method of tf.keras Layer.
        :param x: inputs as tf.Tensor
        :return: policy, value state and probabilities as tf.Tensor
        """

        for l in self.common_body:
            x = l(x)

        policy = self.actor(x)
        probs = tf.nn.softmax(policy)
        value = self.critic(x)

        return policy, value, probs

    def train(self, env, num_steps, render, gamma):
        """
        Training loop.
        :param env: gym environment
        :param num_steps: training steps in the environment as int
        :param render: True if you want to render the environment while training
        :param gamma: discount factor as double
        :return:
        """

        frames = 0

        rewards = []
        actions = []
        states = []

        # Create policy
        policy = StochasticPolicy(env.action_space.n)

        while frames < num_steps:
            game_over = False
            s_t = env.reset()
            score = 0

            while not game_over:

                if render:
                    env.render()

                actor, value, probs = self.call(s_t.reshape(1, *s_t.shape))
                probs = probs.numpy().reshape(-1)
                a_t = policy.select_action(probs)
                action = np.zeros(env.action_space.n)
                action[a_t] = 1
                actions.append(action)
                states.append(s_t)
                s_tp1, r_t, game_over, _ = env.step(a_t)
                rewards.append(r_t)
                s_tp1 = np.array(s_tp1)
                s_t = s_tp1

                score += r_t
                frames += 1

            print('Epochs: {} | Reward: {}'.format(frames, score))

            q_vals = calc_qvals(rewards, gamma=gamma)
            states = np.asarray(states)
            q_vals = np.asarray(q_vals)
            actions = np.asarray(actions)

            with tf.GradientTape(persistent=True) as tape:
                policies, values, probs = self.call(states)
                policy_loss, value_loss, entropy_loss, = \
                    __compute_a2c_loss__(policies, probs, values, q_vals, actions)

            dloss_policy = tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(dloss_policy, self.trainable_variables))

            dloss_value = tape.gradient(value_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(dloss_value, self.trainable_variables))

            dloss_entropy = tape.gradient(entropy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(dloss_entropy, self.trainable_variables))

            states = []
            q_vals = []
            rewards = []
            actions = []

            print('Frame: {}/{} | Score: {} | Loss policy: {} | Loss value: {} | Loss entropy: {}'.
                  format(frames, num_steps, score, policy_loss, value_loss, entropy_loss))


########################################################################################################################


class DDPG:
    """
    Definition of Deep Determistic Policy Gradient model and custom training loop.
    """
    class Critic(tf.keras.layers.Layer):
        def __init__(self, input_shape, hidden_units, output_size):
            """
            Critic model definition.
            :param input_shape: input shape as tuple
            :param hidden_units: number of hidden units for each layer as list.
            :param output_size:  number of actions as integer.
            """
            super(DDPG.Critic, self).__init__()

            # Neural Network definition
            self.layers = []
            for hidden in hidden_units:
                self.layers.append(tf.keras.layers.Dense(units=hidden, activation=tf.nn.relu))
            self.layers.append(tf.keras.layers.Dense(units=output_size))

            # Build TensorFlow graph
            self.build(input_shape)

        def build(self, input_shape):
            """
            Implement build method od tf.keras Layer
            :param input_shape: input shape as tuple
            :return:
            """
            super(DDPG.Critic, self).build(input_shape)

        def call(self, x):
            """
            Implement call method of tf.keras Layer.
            :param x: inputs as tf.Tensor
            :return: Q-values for state x as tf.Tensor
            """

            for l in self.layers:
                x = l(x)

            return x

    ####################################################################################################################

    class Actor(tf.keras.layers.Layer):
        def __init__(self, input_shape, hidden_units, output_size):
            """
            Actor model definition.
            :param input_shape: input shape as tuple.
            :param hidden_units: number of hidden units for each layer as list.
            :param output_size:  number of actions as integer.
            """
            super(DDPG.Actor, self).__init__()

            # Neural Network definition
            self.layers = []
            for hidden in hidden_units:
                self.layers.append(tf.keras.layers.Dense(units=hidden, activation=tf.nn.relu))
            self.layers.append(tf.keras.layers.Dense(units=output_size, activation=tf.nn.tanh))

            # Build TensorFlow graph
            self.build(input_shape)

        def build(self, input_shape):
            """
            Implement build method of tf.keras Layer
            :param input_shape: input shape as tuple
            :return:
            """
            super(DDPG.Actor, self).build(input_shape)

        def call(self, x):
            """
            Implement call method of tf.keras Layer.
            :param x: inputs as tf.Tensor
            :return: Q-value for state x as tf.Tensor
            """

            for l in self.layers:
                x = l(x)

            return x

    ####################################################################################################################

    def __init__(self, num_states, num_actions, actor_hidden_units, critic_hidden_units):
        # Initialize actor and critic
        self.actor = DDPG.Actor(num_states, actor_hidden_units, num_actions)
        self.critic = DDPG.Critic(num_states + num_actions, critic_hidden_units, num_actions)

        self.num_states = num_states
        self.num_actions = num_actions

        # Fake forward step to build the TensorFlow graph
        self.actor(np.zeros(shape=(1, num_states)))
        self.critic(np.zeros(shape=(1, num_states + num_actions)))

        # Initialize target actor and critic
        self.target_actor = DDPG.Actor(num_states, actor_hidden_units, num_actions)
        self.target_critic = DDPG.Critic(num_states, critic_hidden_units, num_actions)
        self.layer_copy(tau=1)
        self.layer_copy(tau=1)

        # Initialize Replay Buffer
        self.memory = ReplayExperienceBuffer(maxlen=50000)

        # Define Adam optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    ####################################################################################################################

    def layer_copy(self, tau):
        for src, tgt in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            tgt.assign(tau * src + (1 - tau) * tgt)

        for src, tgt in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            tgt.assign(tau * src + (1 - tau) * tgt)

    ####################################################################################################################

    def __compute_critic_loss__(self, x, y, actions):
        loss = tf.reduce_mean(tf.square(y - self.critic(np.concatenate((x, actions), axis=1))))

        return loss

    ####################################################################################################################

    def __compute_actor_loss__(self, states):

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_actions = tf.concat([states, self.actor(states)], 1)

        loss = -tf.reduce_mean(self.critic(states_actions))

        return loss

    ####################################################################################################################

    def train(self, env, num_steps, render, gamma, batch_size):
        """
        Training loop.
        :param env: gym environment
        :param num_steps: training steps in the environment as int
        :param render: True if you want to render the environment while training
        :param gamma: discount factor as double
        :return:
        """

        frames = 0

        # Create random noise process
        noise = OUNoise(env.action_space)

        while frames < num_steps:
            game_over = False
            s_t = env.reset()
            score = 0
            noise.reset()

            while not game_over:

                frames += 1

                if render:
                    env.render()

                a_t = self.actor(s_t.reshape((1, self.num_states))).numpy()
                s_tp1, r_t, game_over, _ = env.step(a_t)
                score += r_t
                s_tp1 = np.array(s_tp1.reshape(1, self.num_states))
                s_t = np.array(s_t.reshape(1, self.num_states))
                a_t = np.array(a_t.reshape(1, self.num_actions))
                self.memory.insert((s_t, a_t, s_tp1, r_t, game_over))
                s_t = s_tp1

                if len(self.memory) > batch_size:
                    batch = self.memory.get_random_batch(batch_size)

                    x = np.zeros((len(batch), self.num_states))

                    y = np.zeros((len(batch), self.num_actions))
                    actions = np.zeros(shape=(len(batch), self.num_actions))

                    for i, b in enumerate(batch):
                        state, action, reward, next_state = b[0], b[1], b[3], b[2]
                        next_action = self.target_actor(next_state.reshape(1, self.num_states))
                        q_vals = self.target_critic(np.concatenate((next_state, next_action), axis=1))
                        y[i] = reward + gamma * q_vals
                        x[i] = state
                        actions[i] = action

                    with tf.GradientTape(persistent=True) as tape:
                        critic_loss = self.__compute_critic_loss__(x, y, actions)
                        actor_loss = self.__compute_actor_loss__(x)

                    dloss_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(dloss_critic, self.critic.trainable_variables))

                    dloss_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(dloss_actor, self.actor.trainable_variables))

                self.layer_copy(tau=1e-2)


            print('Frame: {}/{} | Score: {} | Actor loss: {} | Critic loss: {}'.
                  format(frames, num_steps, score, actor_loss, critic_loss))


########################################################################################################################

