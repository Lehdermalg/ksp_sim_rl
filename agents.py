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
        # self.q_network = self._build_q_network(64, 0.01, 0.2, 64, 0.01, 0.2)
        # self.q_network = self._build_q_network(64,0.05,0.2, 64, 0.05, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 64, 0.01, 0.2)
        self.q_network = self._build_q_network(512,0.01,0.2, 512, 0.01, 0.2)
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
