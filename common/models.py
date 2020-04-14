# Author Mattia Silvestri

import tensorflow as tf

import os
import sys
import numpy as np

cwd = os.getcwd()
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '{}/../'.format(cwd))


########################################################################################################################


class DRLModel(tf.keras.Model):
    """
    Deep Reinforcement Learning model.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False):
        """
        Common initialization for all the DRL methods.
        :param output_dim: output dimension of the neural network, i.e. the actions space; as integer
        :param hidden_units: units for each hidden layer; as list of integer
        :param atari: if True, a convolutional architecture is chosen; as boolean
        """
        super(DRLModel, self).__init__()
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

        # Create the actor
        self.actor = tf.keras.layers.Dense(output_dim)

        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam()

    def forward(self, x):
        """
        Implement forward step of tf.keras.Model.
        :param x: inputs; as numpy array
        :return: hidden output; as tf.Tensor
        """

        for l in self.model:
            x = l(x)

        return x

    def act(self, x):
        """
        Given input states, return probability of actions.
        :param x: inputs; as numpy array
        :return: probabilities of actions over states; as tf.Tensor
        """

        x = self.forward(x)
        logits = self.actor(x)

        return tf.nn.softmax(logits)

    def gradient_step(self, *args, **kwargs):
        """
        Compute loss and gradients. Perform an update step.
        :param next_estimated_return: expected return computed with Monte Carlo sampling; as list of float
        :param actions: actions performed at each step; as list of numpy array
        :return: loss values; as list of integer
        """
        raise NotImplementedError("gradient_step() method not implemented")

########################################################################################################################


def __compute_pg_loss__(policies, next_estimated_return, actions):
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=policies, labels=actions)
    policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, next_estimated_return))

    return policy_loss

########################################################################################################################


class PolicyGradient(DRLModel):
    """
        Definition of Policy Gradient RL algorithm.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False):
        """
        :param output_dim: output dimension as integer
        """

        super(PolicyGradient, self).__init__(output_dim, hidden_units, atari)

    def gradient_step(self, states, q_vals, actions):
        """
        Compute loss and gradients. Perform an update step.
        :param states: states of sampled trajectory
        :param q_vals: expected return computed with Monte Carlo sampling; as list of float
        :param actions: actions of sampled trajectory; as list of numpy array
        :return: loss values; as list of integer
        """

        # Tape the gradient during forward step and loss computation
        with tf.GradientTape(persistent=True) as tape:
            x = self.forward(states)
            logits = self.actor(x)
            policy_loss = __compute_pg_loss__(logits, q_vals, actions)

        # Perform un update step
        dloss_policy = tape.gradient(policy_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(dloss_policy, self.trainable_variables))

        return policy_loss

########################################################################################################################


def __compute_a2c_loss__(policies, values, next_estimated_return, actions):
    advantage = next_estimated_return - tf.stop_gradient(values)
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=policies, labels=actions)
    policy_loss = tf.reduce_mean(neg_log_prob * advantage)
    value_loss = tf.reduce_mean((values - next_estimated_return) ** 2)
    # entropy_loss = 0 * tf.reduce_mean(tf.reduce_sum(probs * tf.nn.log_softmax(policies), axis=1))

    return policy_loss, value_loss

########################################################################################################################


class A2C(DRLModel):
    """
        Definition of A2C model and custom training loop.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False):
        """
        :param output_dim: output dimension as integer
        """

        super(A2C, self).__init__(output_dim, hidden_units, atari)

        # Define actor and critic
        self.actor = tf.keras.layers.Dense(output_dim)
        self.critic = tf.keras.layers.Dense(1)

    def gradient_step(self, states, q_vals, actions):
        """
        Compute loss and gradients. Perform an update step.
        :param states: states of sampled trajectory
        :param q_vals: expected return computed with Monte Carlo sampling; as list of float
        :param actions: actions of sampled trajectory; as list of numpy array
        :return: loss values; as list of integer
        """

        with tf.GradientTape(persistent=True) as tape:
            # Forward step of common body
            x = self.forward(states)

            # Compute logits and probs over actions
            logits = self.actor(x)
            probs = tf.nn.softmax(logits)

            # Compute value function
            values = self.critic(x)

            policy_loss, value_loss = \
                __compute_a2c_loss__(logits, values, q_vals, actions)

        dloss_policy = tape.gradient(policy_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(dloss_policy, self.trainable_variables))

        dloss_value = tape.gradient(value_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(dloss_value, self.trainable_variables))

        # dloss_entropy = tape.gradient(entropy_loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(dloss_entropy, self.trainable_variables))

        return policy_loss


########################################################################################################################

class DeepQLearning(DRLModel):
    """
        Deep Q-learning algorithm.
    """

    def __init__(self, output_dim, hidden_units=[32, 32], atari=False, update_interval=1000, tau=0.99):
        """
        Common initialization for all the DRL methods.
        :param output_dim: output dimension of the neural network, i.e. the actions space; as integer
        :param hidden_units: units for each hidden layer; as list of integer
        :param atari: if True, a convolutional architecture is chosen; as boolean
        :param update_interval: target network update interval; as integer
        :param tau: parameter for soft update; as float
        """

        super(DRLModel, self).__init__()
        self.output_dim = output_dim
        self.update_interval = update_interval
        self.tau = tau

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

        # Create the actor
        self.model.append(tf.keras.layers.Dense(output_dim))

        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Create target neural network
        self.tgt_net = []
        if not atari:
            for hidden in hidden_units:
                self.tgt_net.append(tf.keras.layers.Dense(units=hidden, activation='relu'))
        else:
            self.tgt_net.append(tf.keras.layers.Conv2D(filters=32,
                                                     kernel_size=[8, 8],
                                                     strides=[4, 4],
                                                     activation='relu'))
            self.tgt_net.append(tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=[4, 4],
                                                     strides=[2, 2],
                                                     activation='relu'))
            self.tgt_net.append(tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=[3, 3],
                                                     strides=[1, 1],
                                                     activation='relu'))

            self.tgt_net.append(tf.keras.layers.Flatten())
            self.tgt_net.append(tf.keras.layers.Dense(512))

        self.tgt_net.append(tf.keras.layers.Dense(output_dim))
        
        # Keep a count for target network update
        self.count = 0

    def act(self, x):
        """
        Given input states, return probability of actions.
        :param x: inputs; as numpy array
        :return: logits of actions over states; as tf.Tensor
        """

        return self.forward(x)

    def tgt_net_forward(self,x ):
        """
        Forward step of target network.
        :param x: input; as tf.Tensor
        :return: output; as tf.Tensor
        """

        for l in self.tgt_net:
            x = l(x)

        return x

    def update_target_network(self):
        """
        Copy trainable variables of the DL model to the target one.
        :param tau: parameter to make a soft copy; 1 means hard copy;  as integer
        :return:
        """
        for src, tgt in zip(self.model.trainable_variables, self.tgt_net.trainable_variables):
            tgt.assign(self.tau * src + (1 - self.tau) * tgt)

        # Reset counter for update
        self.count = 0

    def gradient_step(self, batch, gamma, update=False):
        """
        Compute loss and gradients. Perform an update step.
        :param batch: sampled batch; as list of tuples
        :param dones: True if it is an end sample, False otherwise; as numpy array
        :return: loss value; as integer
        """

        # Compute (x,y) training pairs
        x, y = self.q_learning(batch, gamma)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        # Compute Q-values and tape gradient
        with tf.GradientTape(persistent=True) as tape:
            q_vals = self.forward(x)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - q_vals)))

        dloss = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(dloss, self.trainable_variables))

        # Update target network if required
        if self.count == self.update_interval:
            self.update_target_network()
            print("Target network updated")

        # Increase counter for target network update
        self.count += 1

        return loss

    def q_learning(self, batch, gamma):
        """
        Q-learning algorithm implementation.
        :param batch: list of samples; as list of tuples
        :param gamma: discount factor; as float
        :return: (x,y) traning instances; as two numpy arrays
        """
        states = np.asarray([val[0] for val in batch])
        next_states = np.asarray([val[3] for val in batch])

        # Predict Q(s,a) given the batch of states
        q_s_a = self.forward(states).numpy()
        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.tgt_net_forward(next_states).numpy()

        # setup training arrays
        x = []
        y = []

        for i, b in enumerate(batch):
            state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
            # Get the current q values for all actions in state
            current_q = q_s_a[i]
            # Update the q value for action
            if done:
                # In this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + gamma * np.amax(q_s_a_d[i])

            x.append(state)
            y.append(current_q)

        return np.asarray(x), np.asarray(y)

########################################################################################################################

'''class DDPG:
    """
    Definition of Deep Determistic Policy Gradient model and custom training loop.
    """
    class Critic(tf.keras.layers.Model):
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

    def __init__(self, input_shape, num_actions, actor_hidden_units, critic_hidden_units):
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
                  format(frames, num_steps, score, actor_loss, critic_loss))'''


########################################################################################################################

