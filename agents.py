import numpy as np
import tensorflow as tf
import logging
import keras
from keras import layers, regularizers, optimizers, callbacks, losses, metrics
from copy import deepcopy

THROTTLE_ACTIONS = np.array([-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20])
ANGLE_ACTIONS = np.array([-5, -2, -1, 0, 1, 2, 5])


class FlightReplayBuffer:
    final_reward = 0.0

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest experience

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def reset(self):
        self.buffer = []


class MultiReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.flight_list = []

    @property
    def find_lowest_reward_flight_index(self) -> int:
        min_reward, index = float("inf"), None
        for i, flight in enumerate(self.flight_list):
            if flight.final_reward < min_reward:
                min_reward = flight.final_reward
                index = i
        return index

    def add(self, flight: FlightReplayBuffer):
        self.flight_list.append(deepcopy(flight))
        if len(self.flight_list) > self.max_size:
            index = self.find_lowest_reward_flight_index
            logging.info(f"Removing lowest reward flight from buffer - reward: {self.flight_list[index].final_reward}")
            self.flight_list.pop(index)  # Remove worst experience

    def sample(self, batch_size):
        _experience_list = []
        for f in self.flight_list:
            _experience_list += f.sample(batch_size)
        return _experience_list


class LossHistory(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss_value = logs['loss']  # Access the loss value
        print(f"Epoch {epoch + 1}: Loss = {loss_value}")


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
        self.q_network = self._build_q_network(network_dict={
            'dense_1': {'n': 128, 'a': 'relu', 'r': 0.01},
            'dropout_1': {'d': 0.2},
            'dense_2': {'n': 64, 'a': 'relu', 'r': 0.01},
            'dropout_2': {'d': 0.2},
            'dense_3': {'n': 128, 'a': 'relu', 'r': 0.01},
            'dropout_3': {'d': 0.2},
        })
        # self.q_network = self._build_q_network(128,0.01,0.2, 128, 0.01, 0.2)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = losses.categorical_crossentropy
        self.metric = metrics.categorical_accuracy

        self.q_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])

        self.q_network.predict(np.zeros((1,) + self.env.observation_space.shape))

    def action_to_index(self, action: list):
        """Converts an action (list) to an action index (list)."""
        throttle_idx = np.where(THROTTLE_ACTIONS * self.env.dt == action[0])[0][0]
        angle_idx = np.where(ANGLE_ACTIONS * self.env.dt == action[1])[0][0]
        action_index = throttle_idx * len(ANGLE_ACTIONS) + angle_idx
        return action_index

    @staticmethod
    def _create_layers_from_dict(layer_dict, keras_layers):
        """
        Iterates through a dictionary defining neural network layers and creates corresponding Keras layer objects.

        Args:
            layer_dict: A dictionary where keys represent layer names and values are dictionaries containing layer parameters.

        Returns:
            A list of Keras layer objects.
        """
        for layer_name, layer_params in layer_dict.items():
            if 'dense' in layer_name:  # Dense layer
                units = layer_params['n']
                activation = layer_params['a']
                regularizer = None
                if 'r' in layer_params:
                    regularizer = regularizers.l2(layer_params['r'])

                dense_layer = layers.Dense(units, activation=activation, kernel_regularizer=regularizer)
                keras_layers.append(dense_layer)

            if 'dropout' in layer_params:  # Dropout layer
                rate = layer_params['d']
                dropout_layer = layers.Dropout(rate)
                keras_layers.append(dropout_layer)

        return keras_layers

    def _build_q_network(self, **network_dict):
        """Creates the neural network for Q-value approximation."""
        layers_list = [layers.Input(shape=self.env.observation_space.shape)]
        layers_list = self._create_layers_from_dict(network_dict, layers_list)
        layers_list.append(layers.Dense(len(THROTTLE_ACTIONS) * len(ANGLE_ACTIONS), activation='linear'))
        model = keras.Sequential(layers_list)
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
            action_index = np.argmax(self.q_values)  # Directly get the index of the best action

            # Convert action_index back to throttle and angle indices
            throttle_idx = action_index // len(ANGLE_ACTIONS)
            angle_idx = action_index % len(ANGLE_ACTIONS)

        # Retrieve the actual throttle and angle values from the arrays
        throttle_delta = THROTTLE_ACTIONS[throttle_idx] * self.env.dt
        thrust_angle_delta = ANGLE_ACTIONS[angle_idx] * self.env.dt

        # print(f"Throttle delta, thrust angle delta: {throttle_delta}, {thrust_angle_delta}")
        return [throttle_delta, thrust_angle_delta]

    def _calculate_target_q_value(self, state, action, reward, next_state, done):
        # Get Q-values for the current state
        q_values = self.q_network.predict(state[np.newaxis, :])[0]

        # Calculate the target Q-value
        if done and next_state.size == 0:
            target = reward
        else:
            next_q_values = self.q_network.predict(next_state[np.newaxis, :])[0]
            best_next_action = np.argmax(next_q_values)
            target = reward + self.gamma * next_q_values[best_next_action]

        # Update the Q-value for the chosen action
        action_index = self.action_to_index(action)
        target_q_values = q_values.copy()  # Create a copy to modify
        target_q_values[action_index] = target

        return target_q_values


    def update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        from environment import OBSERVATION_NAMES

        target_q_values = self._calculate_target_q_value(state, action, reward, next_state, done)

        # Prepare data for training
        x = state[np.newaxis, :]  # Input state
        y = target_q_values[np.newaxis, :]  # Target Q-values

        # Perform a training step using model.fit (enables callbacks)
        history = self.q_network.fit(x, y, epochs=1, verbose=0)  # Single epoch, no output

        # Access loss from the history object
        loss = history.history['loss'][0]

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

    def prepare_q_values_for_post_episode_fitting(self, batch):
        # Calculate target Q-values for each state-action pair in the batch
        target_q_values = []
        for state, action, reward, next_state, done in batch:
            target_q_values_for_state = self._calculate_target_q_value(state, action, reward, next_state, done)
            target_q_values.append(target_q_values_for_state)

        y = np.array(target_q_values)  # Target Q-values (2D array)

        return y
