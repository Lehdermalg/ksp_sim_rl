import numpy as np
import tensorflow as tf
import logging
import keras
from keras import layers, regularizers, optimizers, callbacks

THROTTLE_ACTIONS = np.array([-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20])
ANGLE_ACTIONS = np.array([-5, -2, -1, 0, 1, 2, 5])


class CustomCallback(callbacks.Callback):
    @staticmethod
    def on_batch_end(batch, rewards):
        if batch % 10 == 0:
            logging.info(f"Batch {batch}: Rewards = {rewards[-1]:.4f}")


class QLearningAgentANN(object):
    def __init__(
            self,
            env,
            writer,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
    ):
        self.env = env
        self.writer = writer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_values = None

        # Build the neural network (you can customize the architecture)
        self.q_network = self._build_q_network(64, 0.01, 0.2, 64, 0.01, 0.2)
        # self.q_network = self._build_q_network(64,0.05,0.2, 64, 0.05, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 64, 0.01, 0.2)
        # This one occasionally gave 180-250s flights but no stable ones
        # self.q_network = self._build_q_network(512,0.01,0.2, 512, 0.01, 0.2)
        # self.q_network = self._build_q_network(1024,0.02,0.2, 1024, 0.02, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 128, 0.01, 0.2)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        self.q_network.predict(np.zeros((1,) + self.env.observation_space.shape))

    def _build_q_network(self, neurons_1, regularization_1, dropout_1, neurons_2, regularization_2, dropout_2):
        """Creates the neural network for Q-value approximation."""
        model = keras.Sequential([
            layers.Input(shape=self.env.observation_space.shape),
            layers.Dense(neurons_1, activation='relu', kernel_regularizer=regularizers.l2(regularization_1)),
            layers.Dropout(dropout_1),  # Add a dropout layer after the dense layer
            layers.Dense(neurons_2, activation='relu', kernel_regularizer=regularizers.l2(regularization_2)),
            layers.Dropout(dropout_2),  # Add a dropout layer after the dense layer
            # Output: Q-values for each action
            # Sigmoid to correctly map to the action space
            # layers.Dense(self.env.action_space.shape[0], activation='sigmoid')
            layers.Dense(len(THROTTLE_ACTIONS) + len(ANGLE_ACTIONS), activation='linear')  # TODO: Why linear?
        ])
        logging.info(f"Action space shape: {self.env.action_space}")
        return model

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Randomly choose throttle and angle actions
            throttle_idx = np.random.choice(len(THROTTLE_ACTIONS))
            angle_idx = np.random.choice(len(ANGLE_ACTIONS))
        else:
            # Get Q-values from the network
            self.q_values = self.q_network.predict(state[np.newaxis, :])[0]
            # Get the indices of the actions with the highest Q-values
            throttle_idx = np.argmax(self.q_values[:len(THROTTLE_ACTIONS)])
            angle_idx = np.argmax(self.q_values[len(THROTTLE_ACTIONS):])
            # Retrieve the actual throttle and angle values from the arrays
        throttle_delta = THROTTLE_ACTIONS[throttle_idx] * self.env.dt
        thrust_angle_delta = ANGLE_ACTIONS[angle_idx] * self.env.dt

        # print(f"Throttle delta, thrust angle delta: {throttle_delta}, {thrust_angle_delta}")
        return [throttle_delta, thrust_angle_delta]

    def update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        from environment import OBSERVATION_NAMES

        with tf.GradientTape() as tape:
            self.q_values = self.q_network(state[np.newaxis, :])
            # print(f"action: {action}")
            # --- Get indices of the chosen actions ---
            throttle_idx = np.where(THROTTLE_ACTIONS*self.env.dt == action[0])[0][0]
            angle_idx = np.where(ANGLE_ACTIONS*self.env.dt == action[1])[0][0]
            # --- Access Q-values using the indices ---
            q_value = self.q_values[0, throttle_idx] + self.q_values[0, len(THROTTLE_ACTIONS) + angle_idx]
            q_value = tf.reduce_sum(q_value)  # Convert to a scalar

            if done and next_state.size == 0:   # Handle the crash
                target = reward
            else:
                next_q_values = self.q_network(next_state[np.newaxis, :])

                # --- Get indices of the best actions in the next state ---
                next_throttle_idx = np.argmax(next_q_values[0, :len(THROTTLE_ACTIONS)])
                next_angle_idx = np.argmax(next_q_values[0, len(THROTTLE_ACTIONS):])

                # --- Calculate the target Q-value for the chosen action ---
                target = reward + self.gamma * (next_q_values[0, next_throttle_idx] +
                                                next_q_values[0, len(THROTTLE_ACTIONS) + next_angle_idx])

            # --- Calculate MSE directly ---
            loss = tf.square(target - q_value)  # No need for tf.reduce_mean()

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        # print(f"Grads: {grads}")
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # --- Get weights from the first Dense layer ---
        first_dense_layer = self.q_network.layers[0]
        # print(f"weights: {first_dense_layer.get_weights()}")
        weights = first_dense_layer.get_weights()[0]

        # --- Calculate normalized sum of squared weights for each observation ---
        squared_weights = np.square(weights)
        sum_squared_weights = np.sum(squared_weights, axis=1)  # Sum along neuron axis
        normalized_importance = sum_squared_weights / len(weights)  # Normalize

        # --- Log normalized importance ---
        with self.writer.as_default():
            tf.summary.histogram('Observation Importance', normalized_importance, step=step)
            for i, observation_name in enumerate(OBSERVATION_NAMES):
                tf.summary.scalar(f'Importance_{observation_name}', normalized_importance[i], step=step)

        return loss

class QLearningAgentANNDouble(object):
    def __init__(
            self,
            env,
            writer,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
    ):
        self.env = env
        self.writer = writer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_values_t = None
        self.q_values_a = None

        # Build the neural network (you can customize the architecture)
        self.q_network_t = self._build_q_network(
            neurons_1=64, regularization_1=0.01, dropout_1=0.2,
            neurons_2=64, regularization_2=0.01, dropout_2=0.2,
            neurons_3=64, regularization_3=0.01, dropout_3=0.2,
            output_actions=THROTTLE_ACTIONS)
        self.q_network_a = self._build_q_network(
            neurons_1=64, regularization_1=0.01, dropout_1=0.2,
            neurons_2=64, regularization_2=0.01, dropout_2=0.2,
            neurons_3=64, regularization_3=0.01, dropout_3=0.2,
            output_actions=ANGLE_ACTIONS)
        # self.q_network = self._build_q_network(64,0.05,0.2, 64, 0.05, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 64, 0.01, 0.2)
        # This one occasionally gave 180-250s flights but no stable ones
        # self.q_network = self._build_q_network(512,0.01,0.2, 512, 0.01, 0.2)
        # self.q_network = self._build_q_network(1024,0.02,0.2, 1024, 0.02, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 128, 0.01, 0.2)
        self.optimizer_t = optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer_a = optimizers.Adam(learning_rate=self.learning_rate)

        self.q_network_t.predict(np.zeros((1,) + self.env.observation_space.shape))
        self.q_network_a.predict(np.zeros((1,) + self.env.observation_space.shape))

    def _build_q_network(
            self,
            output_actions,
            neurons_1, regularization_1, dropout_1 = None,
            neurons_2 = None, regularization_2 = None, dropout_2 = None,
            neurons_3 = None, regularization_3 = None, dropout_3 = None,
            neurons_4 = None, regularization_4 = None, dropout_4 = None,
        ):
        """Creates the neural network for Q-value approximation."""
        layer_list = [
            layers.Input(shape=self.env.observation_space.shape),
            layers.Dense(neurons_1, activation='relu', kernel_regularizer=regularizers.l2(regularization_1))
        ]
        if dropout_1 is not None:
            layer_list.append(
                layers.Dropout(dropout_1))

        if neurons_2 is not None:
            layer_list.append(
                layers.Dense(neurons_2, activation='relu', kernel_regularizer=regularizers.l2(regularization_2)))
        if dropout_2 is not None:
            layer_list.append(
                layers.Dropout(dropout_2))

        if neurons_3 is not None:
            layer_list.append(
                layers.Dense(neurons_3, activation='relu', kernel_regularizer=regularizers.l2(regularization_3)))
        if dropout_3 is not None:
            layer_list.append(
                layers.Dropout(dropout_3))

        if neurons_4 is not None:
            layer_list.append(
                layers.Dense(neurons_4, activation='relu', kernel_regularizer=regularizers.l2(regularization_4)))
        if dropout_4 is not None:
            layer_list.append(
                layers.Dropout(dropout_4))

        layer_list.append(layers.Dense(len(output_actions), activation='linear'))

        model = keras.Sequential(layer_list)
        if output_actions is THROTTLE_ACTIONS:
            logging.info(f"Action space shape: {self.env.action_space_t}")
        if output_actions is ANGLE_ACTIONS:
            logging.info(f"Action space shape: {self.env.action_space_a}")
        return model

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Randomly choose throttle and angle actions
            throttle_idx = np.random.choice(len(THROTTLE_ACTIONS))
            angle_idx = np.random.choice(len(ANGLE_ACTIONS))
        else:
            # Get Q-values from the network
            self.q_values_t = self.q_network_t.predict(state[np.newaxis, :])[0]
            self.q_values_a = self.q_network_a.predict(state[np.newaxis, :])[0]
            # Get the indices of the actions with the highest Q-values
            throttle_idx = np.argmax(self.q_values_t[:len(THROTTLE_ACTIONS)])
            angle_idx = np.argmax(self.q_values_a[:len(ANGLE_ACTIONS)])
            # Retrieve the actual throttle and angle values from the arrays
        throttle_delta = THROTTLE_ACTIONS[throttle_idx] * self.env.dt
        thrust_angle_delta = ANGLE_ACTIONS[angle_idx] * self.env.dt

        # print(f"Throttle delta, thrust angle delta: {throttle_delta}, {thrust_angle_delta}")
        return [throttle_delta, thrust_angle_delta]

    def update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        from environment import OBSERVATION_NAMES

        with tf.GradientTape() as tape_t:
            self.q_values_t = self.q_network_t(state[np.newaxis, :])
            # print(f"action: {action}")
            # --- Get indices of the chosen actions ---
            throttle_idx = np.where(THROTTLE_ACTIONS*self.env.dt == action[0])[0][0]
            # --- Access Q-values using the indices ---
            q_value_t = self.q_values_t[0, throttle_idx]
            q_value_t = tf.reduce_sum(q_value_t)  # Convert to a scalar

            if done and next_state.size == 0:   # Handle the crash
                target_t = reward
            else:
                next_q_values_t = self.q_network_t(next_state[np.newaxis, :])

                # --- Get indices of the best actions in the next state ---
                next_throttle_idx = np.argmax(next_q_values_t[0, :len(THROTTLE_ACTIONS)])

                # --- Calculate the target Q-value for the chosen action ---
                target_t = reward + self.gamma * next_q_values_t[0, next_throttle_idx]

            # --- Calculate MSE directly ---
            loss_t = tf.square(target_t - q_value_t)  # No need for tf.reduce_mean()

        with tf.GradientTape() as tape_a:
            self.q_values_a = self.q_network_a(state[np.newaxis, :])
            # print(f"action: {action}")
            # --- Get indices of the chosen actions ---
            angle_idx = np.where(ANGLE_ACTIONS*self.env.dt == action[1])[0][0]
            # --- Access Q-values using the indices ---
            q_value_a = self.q_values_a[0, angle_idx]
            q_value_a = tf.reduce_sum(q_value_a)  # Convert to a scalar

            if done and next_state.size == 0:   # Handle the crash
                target_a = reward
            else:
                next_q_values_a = self.q_network_a(next_state[np.newaxis, :])

                # --- Get indices of the best actions in the next state ---
                next_angle_idx = np.argmax(next_q_values_a[0, :len(ANGLE_ACTIONS)])

                # --- Calculate the target Q-value for the chosen action ---
                target_a = reward + self.gamma * next_q_values_a[0, next_angle_idx]

            # --- Calculate MSE directly ---
            loss_a = tf.square(target_a - q_value_a)  # No need for tf.reduce_mean()

        target = reward + self.gamma * (next_q_values_t[0, next_throttle_idx] +
                                        next_q_values_a[0, next_angle_idx])
        loss = tf.square(target - q_value_t - q_value_a)  # No need for tf.reduce_mean()

        grads_t = tape_t.gradient(loss_t, self.q_network_t.trainable_variables)
        grads_a = tape_a.gradient(loss_a, self.q_network_a.trainable_variables)
        # print(f"Grads: {grads}")
        self.optimizer_t.apply_gradients(zip(grads_t, self.q_network_t.trainable_variables))
        self.optimizer_a.apply_gradients(zip(grads_a, self.q_network_a.trainable_variables))

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # --- Get weights from the first Dense layer ---
        first_dense_layer_t = self.q_network_t.layers[0]
        first_dense_layer_a = self.q_network_a.layers[0]
        # print(f"weights: {first_dense_layer.get_weights()}")
        weights_t = first_dense_layer_t.get_weights()[0]
        weights_a = first_dense_layer_a.get_weights()[0]

        # --- Calculate normalized sum of squared weights for each observation ---
        squared_weights_t = np.square(weights_t)
        squared_weights_a = np.square(weights_a)
        sum_squared_weights_t = np.sum(squared_weights_t, axis=1)  # Sum along neuron axis
        sum_squared_weights_a = np.sum(squared_weights_a, axis=1)  # Sum along neuron axis
        normalized_importance_t = sum_squared_weights_t / len(weights_t)  # Normalize
        normalized_importance_a = sum_squared_weights_a / len(weights_a)  # Normalize

        # --- Log normalized importance ---
        with self.writer.as_default():
            tf.summary.histogram('Observation Importance T', normalized_importance_t, step=step)
            tf.summary.histogram('Observation Importance A', normalized_importance_a, step=step)
            for i, observation_name in enumerate(OBSERVATION_NAMES):
                tf.summary.scalar(f'Importance_{observation_name}', normalized_importance_t[i], step=step)
                tf.summary.scalar(f'Importance_{observation_name}', normalized_importance_a[i], step=step)

        return loss, loss_t, loss_a
