import numpy as np
import tensorflow as tf
import logging
import keras
import json
from keras import layers, regularizers, optimizers, callbacks, losses, metrics
from copy import deepcopy

THROTTLE_ACTIONS = np.array([-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20])
ANGLE_ACTIONS = np.array([-5, -2, -1, 0, 1, 2, 5])
T, A = np.meshgrid(THROTTLE_ACTIONS, ANGLE_ACTIONS)
ALL_ACTIONS = list(zip(T.flatten(), A.flatten()))

buffer_folder = "replay_buffers"
experience_folder = "experience_buffers"


class FlightReplayBuffer:
    final_reward = 0.0

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest experience

    def sample(self, batch_size: int, n_th: int = 100):
        """"""
        assert isinstance(n_th, int), f'n_th must be an integer. {n_th}'
        assert n_th > 0, f'n_th must be larger than 0. {n_th}'

        # print(f"Buffer size: {len(self.buffer)//n_th}. Batch_size: {batch_size}")
        # print(f"min: {min(len(self.buffer)//n_th, batch_size)}")
        indices = np.random.choice(
            len(self.buffer)//n_th,
            min(len(self.buffer)//n_th, batch_size),
            replace=False
        )
        _sub_buffer = self.buffer[::n_th]
        return [_sub_buffer[i] for i in indices]

    def reset(self):
        self.buffer = []

    def save(self, filename):
        """Saves the replay buffer to a JSON file."""
        data = {
            'max_size': self.max_size,
            'total_reward': self.final_reward,
            'experiences':  [(s.tolist(), a.tolist(), r, ns.tolist(), d) for s, a, r, ns, d in self.buffer]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename):
        """Loads the replay buffer from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.max_size = data['max_size']
        self.final_reward = data['total_reward']
        self.buffer = [(np.array(s), np.array(a), r, np.array(ns), d) for s, a, r, ns, d in data['experiences']]


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

    def sample(self, batch_size, n_th: int = 100):
        assert isinstance(n_th, int), f'n_th must be an integer. {n_th}'
        assert n_th > 0, f'n_th must be larger than 0. {n_th}'

        _experience_list = []
        for f in self.flight_list:
            # print(f"Batch size: {batch_size}. Buffer size: {len(f.buffer)}. Nth: {n_th}")
            _experience_list += f.sample(
                min(batch_size, len(f.buffer)),
                n_th=n_th
            )

        logging.info(f"Prepared an experience sample of {len(_experience_list)} length using every {n_th} point.")
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
    epsilon_restart = None
    small_epochs = 1
    _layers = []

    def __init__(
            self,
            env,
            writer,
            **agent_params
    ):
        self.env = env
        self.writer = writer

        self.learning_rate = agent_params['learning_rate']
        self.gamma = agent_params['gamma']
        self.epsilon = agent_params['epsilon_hi']
        self.max_epsilon = agent_params['epsilon_hi']
        self.min_epsilon = agent_params['epsilon_lo']
        self.epsilon_decay = agent_params['epsilon_decay']
        self.flights_recorded = agent_params['flights_recorded']
        self.flight_steps_recorded = agent_params['flight_steps_recorded']
        self.q_values = None

        # Create a SINGLE FLIGHT replay buffer
        self.flight_replay_buffer = FlightReplayBuffer(max_size=self.flight_steps_recorded)  # Adjust max_size as needed

        # Create a MULTI FLIGHT replay buffer
        self.multi_replay_buffer = MultiReplayBuffer(max_size=self.flights_recorded)  # Adjust max_size as needed

        # Build the neural network (you can customize the architecture)
        self.q_network = self._build_q_network(network_dict={
            'dense_1': {'n': 64, 'a': 'relu', 'r': 0.01},
            'dropout_1': {'d': 0.2},
            'dense_2': {'n': 32, 'a': 'relu', 'r': 0.01},
            'dropout_2': {'d': 0.2},
            'dense_3': {'n': 64, 'a': 'relu', 'r': 0.01},
            'dropout_3': {'d': 0.2},
        })

        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # self.loss = losses.categorical_crossentropy
        # self.metric = metrics.categorical_accuracy
        self.loss = losses.sparse_categorical_crossentropy
        self.metric = metrics.sparse_categorical_accuracy

        self.q_network.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])

        # show me what you've got...
        self.q_network.summary()
        # keras.utils.plot_model(self.q_network, to_file='model.png', show_shapes=True)

        self.q_network.predict(np.zeros((1,) + self.env.observation_space.shape))

    def action_to_index(self, action: list):
        """Converts an action (list) to an action index (list)."""
        throttle_idx = np.where(THROTTLE_ACTIONS * self.env.dt == action[0])[0][0]
        angle_idx = np.where(ANGLE_ACTIONS * self.env.dt == action[1])[0][0]
        action_index = throttle_idx * len(ANGLE_ACTIONS) + angle_idx
        return action_index

    def _create_layers_from_dict(self, layer_dict):
        """
        Iterates through a dictionary defining neural network layers and creates corresponding Keras layer objects.

        Args:
            layer_dict: A dictionary where keys represent layer names and values are dictionaries containing layer parameters.

        Returns:
            A list of Keras layer objects.
        """
        self._layers = [layers.Input(shape=self.env.observation_space.shape)]

        for layer_name, layer_params in layer_dict.items():
            # print(f"Browsing layer dict item: {layer_name}, {layer_params}")
            if 'dense' in layer_name:  # Dense layer
                # print(f"Building dense layer")
                units = layer_params['n']
                activation = layer_params['a']
                regularizer = None
                if 'r' in layer_params:
                    regularizer = regularizers.l2(layer_params['r'])

                dense_layer = layers.Dense(units, activation=activation, kernel_regularizer=regularizer)
                self._layers.append(dense_layer)

            if 'dropout' in layer_name:  # Dropout layer
                # print(f"Building dropout layer")
                rate = layer_params['d']
                dropout_layer = layers.Dropout(rate)
                self._layers.append(dropout_layer)

    def _build_q_network(self, network_dict: dict):
        """Creates the neural network for Q-value approximation."""
        self._create_layers_from_dict(network_dict)
        # print(f"Layers: {self._layers}")
        # layers_list.append(layers.Dense(len(THROTTLE_ACTIONS) * len(ANGLE_ACTIONS), activation='linear'))
        self._layers.append(layers.Dense(len(THROTTLE_ACTIONS) * len(ANGLE_ACTIONS), activation='softmax'))
        # print(f"Layers: {self._layers}")
        model = keras.Sequential(self._layers)
        # print(f"Model: {model}")
        logging.info(f"Action space shape: {self.env.action_space}")
        return model

    def choose_action(self, state) -> np.array:
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Randomly choose throttle and angle actions
            throttle_idx = np.random.choice(len(THROTTLE_ACTIONS))
            angle_idx = np.random.choice(len(ANGLE_ACTIONS))
            logging.info(f"Choosing random: {throttle_idx}, {angle_idx} == "
                         f"{THROTTLE_ACTIONS[throttle_idx]}, {ANGLE_ACTIONS[angle_idx]}")
        else:
            # Get Q-values from the network
            self.q_values = self.q_network.predict(state[np.newaxis, :])[0]
            # print(f"Q values: {self.q_values}")
            action_index = np.argmax(self.q_values)

            # Convert action_index back to throttle and angle indices
            throttle_idx = action_index // len(ANGLE_ACTIONS)
            angle_idx = action_index % len(ANGLE_ACTIONS)
            logging.info(f"Action index: {action_index}, "
                         f"Value: {self.q_values[action_index]}, "
                         f"Throttle: {THROTTLE_ACTIONS[throttle_idx]}, "
                         f"Angle: {ANGLE_ACTIONS[angle_idx]}")

        # Retrieve the actual throttle and angle values from the arrays
        throttle_delta = THROTTLE_ACTIONS[throttle_idx] * self.env.dt
        thrust_angle_delta = ANGLE_ACTIONS[angle_idx] * self.env.dt

        # print(f"Throttle delta, thrust angle delta: {throttle_delta}, {thrust_angle_delta}")
        return np.array([throttle_delta, thrust_angle_delta])

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
        """Updates the Q-network using sparse categorical cross-entropy loss and Keras training."""
        from environment import OBSERVATION_NAMES

        # Convert action to an index
        action_index = self.action_to_index(action)

        # Prepare data for training
        x = state[np.newaxis, :]  # Input state
        y = np.array([action_index])  # Target action index (1D array)

        # Perform a training step using model.fit (enables callbacks)
        history = self.q_network.fit(x, y, epochs=self.small_epochs, verbose=1)

        # Access loss from the history object
        loss = history.history['loss'][0]

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # --- Get weights from the first Dense layer ---
        first_dense_layer = self.q_network.layers[0]
        # print(f"weights: {first_dense_layer.get_weights()}")
        weights = first_dense_layer.get_weights()[0]
        weights_l = self.q_network.layers[-1].get_weights()[0]

        # --- Calculate normalized sum of squared weights for each observation ---
        squared_weights = np.square(weights)
        sum_squared_weights = np.sum(squared_weights, axis=1)  # Sum along neuron axis
        normalized_importance = sum_squared_weights / len(weights)  # Normalize

        # --- Calculate normalized sum of squared weights for each observation ---
        squared_weights_l = np.square(weights_l)
        sum_squared_weights_l = np.sum(squared_weights_l, axis=0)  # Sum along neuron axis
        normalized_importance_l = sum_squared_weights_l / len(weights_l)  # Normalize

        # --- Log normalized importance ---
        with self.writer.as_default():
            tf.summary.histogram('Observation Importance', normalized_importance, step=step)
            tf.summary.histogram('Action Importance', normalized_importance_l, step=step)
            for i, observation_name in enumerate(OBSERVATION_NAMES):
                tf.summary.scalar(f'Importance_{observation_name}', normalized_importance[i], step=step)
            for i, action in enumerate(ALL_ACTIONS):
                tf.summary.scalar(f'Importance_T{action[0]}A{action[1]}', normalized_importance_l[i], step=step)

        return loss

    def old_update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        from environment import OBSERVATION_NAMES

        target_q_values = self._calculate_target_q_value(state, action, reward, next_state, done)

        # Prepare data for training
        x = state[np.newaxis, :]  # Input state
        y = target_q_values[np.newaxis, :]  # Target Q-values

        # Perform a training step using model.fit (enables callbacks)
        history = self.q_network.fit(x, y, epochs=self.small_epochs, verbose=1)

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

    def prepare_target_action_indices(self, batch):
        # Calculate target action indices for each state-action pair in the batch
        target_action_indices = []
        for state, action, reward, next_state, done in batch:
            target_q_values_for_state = self._calculate_target_q_value(state, action, reward, next_state, done)
            best_action_index = np.argmax(target_q_values_for_state)  # Get the index of the best action
            target_action_indices.append(best_action_index)

        y = np.array(target_action_indices)  # Target action indices (1D array)
        return y

    def prepare_q_values_for_post_episode_fitting(self, batch):
        # Calculate target Q-values for each state-action pair in the batch
        target_q_values = []
        for state, action, reward, next_state, done in batch:
            target_q_values_for_state = self._calculate_target_q_value(state, action, reward, next_state, done)
            target_q_values.append(target_q_values_for_state)

        y = np.array(target_q_values)  # Target Q-values (2D array)

        return y

    def save_or_drop_flight_buffer(self, final_reward, filename):
        self.flight_replay_buffer.final_reward = final_reward
        if self.flight_replay_buffer.final_reward > 0:
            self.flight_replay_buffer.save(filename)
            logging.info(f"Adding new flight to buffer "
                         f"- size {len(self.multi_replay_buffer.flight_list)} "
                         f"- reward: {self.flight_replay_buffer.final_reward}")
            self.multi_replay_buffer.add(self.flight_replay_buffer)
        self.flight_replay_buffer.reset()

    def load_experience_from_folder(self, replay_folder):
        import glob, os
        logging.info(f"Loading experience from folder: {replay_folder}")
        # --- Load Experience Replay Files from a Folder ---
        replay_filenames = glob.glob(os.path.join(replay_folder, "*.json"))  # Get all JSON files in the folder
        if len(replay_filenames) > self.multi_replay_buffer.max_size:
            logging.warning(f"Amount of flights available was {len(replay_filenames)} > "
                            f"flight buffer size {self.multi_replay_buffer.max_size}. "
                            f"Some flights will be dropped!")
        for filename in replay_filenames:
            self.flight_replay_buffer.reset()
            self.flight_replay_buffer.load(filename)
            self.multi_replay_buffer.add(self.flight_replay_buffer)

    def train_on_experience(self, large_epochs: int, small_epochs: int, batch_size: int):
        """Trains the agent on previously stored experience.
            large_epochs: how many times the agent will get trained on batches of data
            small_epochs: how many times the agent will get trained on a single batch sample
            batch_size: how many samples to take from each batch
        """
        logging.info(f"Training on experience: "
                     f"{large_epochs} x "
                     f"{small_epochs} x "
                     f"{batch_size} x "
                     f"{len(self.multi_replay_buffer.flight_list)}")
        for epoch in range(large_epochs):
            batch = self.multi_replay_buffer.sample(batch_size)
            logging.info(f"Epoch {epoch + 1}/{large_epochs}")
            if len(batch) > 0:
                states, actions, rewards, next_states, dones = zip(*batch)

                x = np.array(states)
                # y = self.prepare_q_values_for_post_episode_fitting(batch)
                y = self.prepare_target_action_indices(batch)

                # Train for one epoch
                history = self.q_network.fit(x, y, epochs=small_epochs, verbose=1)

                # Log loss (if needed)
                loss = history.history['loss'][0]
                logging.info(f"Pre-training Epoch {epoch + 1}: Loss = {loss:.4f}")


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
            network_dict={
                'dense_1': {'n': 64, 'a': 'relu', 'r': 0.01}, 'dropout_1': {'d': 0.2},
                'dense_2': {'n': 32, 'a': 'relu', 'r': 0.01}, 'dropout_2': {'d': 0.2},
                'dense_3': {'n': 64, 'a': 'relu', 'r': 0.01}, 'dropout_3': {'d': 0.2},
            },
            output_actions=THROTTLE_ACTIONS)
        self.q_network_a = self._build_q_network(
            network_dict={
                'dense_1': {'n': 64, 'a': 'relu', 'r': 0.01}, 'dropout_1': {'d': 0.2},
                'dense_2': {'n': 32, 'a': 'relu', 'r': 0.01}, 'dropout_2': {'d': 0.2},
                'dense_3': {'n': 64, 'a': 'relu', 'r': 0.01}, 'dropout_3': {'d': 0.2},
            },
            output_actions=ANGLE_ACTIONS)
        # self.q_network = self._build_q_network(64,0.05,0.2, 64, 0.05, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 64, 0.01, 0.2)
        # This one occasionally gave 180-250s flights but no stable ones
        # self.q_network = self._build_q_network(512,0.01,0.2, 512, 0.01, 0.2)
        # self.q_network = self._build_q_network(1024,0.02,0.2, 1024, 0.02, 0.2)
        # self.q_network = self._build_q_network(128,0.01,0.2, 128, 0.01, 0.2)
        self.optimizer_t = optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer_a = optimizers.Adam(learning_rate=self.learning_rate)

        self.loss_t = losses.categorical_crossentropy
        self.loss_a = losses.categorical_crossentropy
        self.metric_t = metrics.categorical_accuracy
        self.metric_a = metrics.categorical_accuracy

        self.q_network_t.compile(optimizer=self.optimizer_t, loss=self.loss_t, metrics=[self.metric_t])
        self.q_network_a.compile(optimizer=self.optimizer_a, loss=self.loss_a, metrics=[self.metric_a])

        self.q_network_t.predict(np.zeros((1,) + self.env.observation_space.shape))
        self.q_network_a.predict(np.zeros((1,) + self.env.observation_space.shape))

    def _build_q_network(
            self,
            output_actions,
            **network_dict
        ):
        """Creates the neural network for Q-value approximation."""
        layers_list = [layers.Input(shape=self.env.observation_space.shape)]
        layers_list = self._create_layers_from_dict(network_dict, layers_list)

        layers_list.append(layers.Dense(len(output_actions), activation='linear'))

        model = keras.Sequential(layers_list)
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

    def old_update(self, state, action, reward, next_state, done, step):
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

    def update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        from environment import OBSERVATION_NAMES

        target_q_values_t, target_q_values_a = self._calculate_target_q_value(state, action, reward, next_state, done)

        # Prepare data for training
        x = state[np.newaxis, :]  # Input state
        t = target_q_values_t[np.newaxis, :]  # Target Q-values
        a = target_q_values_a[np.newaxis, :]  # Target Q-values

        # Perform a training step using model.fit (enables callbacks)
        history_t = self.q_network_t.fit(x, t, epochs=1, verbose=0)  # Single epoch, no output
        history_a = self.q_network_a.fit(x, a, epochs=1, verbose=0)  # Single epoch, no output

        # Access loss from the history object
        loss_t = history_t.history['loss'][0]
        loss_a = history_a.history['loss'][0]

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
                tf.summary.scalar(f'T_Importance_{observation_name}', normalized_importance_t[i], step=step)
                tf.summary.scalar(f'A_Importance_{observation_name}', normalized_importance_a[i], step=step)

        return loss_t, loss_a

    def action_to_index(self, action: list):
        """Converts an action (list) to an action index (list)."""
        throttle_idx = np.where(THROTTLE_ACTIONS * self.env.dt == action[0])[0][0]
        angle_idx = np.where(ANGLE_ACTIONS * self.env.dt == action[1])[0][0]
        action_index = throttle_idx, angle_idx
        return action_index

    def _calculate_target_q_value(self, state, action, reward, next_state, done):
        # Get Q-values for the current state
        q_values_t = self.q_network_t.predict(state[np.newaxis, :])[0]
        q_values_a = self.q_network_a.predict(state[np.newaxis, :])[0]

        # Calculate the target Q-value
        if done and next_state.size == 0:
            target_t = reward
            target_a = reward
        else:
            next_q_values_t = self.q_network_t.predict(next_state[np.newaxis, :])[0]
            next_q_values_a = self.q_network_a.predict(next_state[np.newaxis, :])[0]
            best_next_action_t = np.argmax(next_q_values_t)
            best_next_action_a = np.argmax(next_q_values_a)
            target_t = reward + self.gamma * next_q_values_t[best_next_action_t]
            target_a = reward + self.gamma * next_q_values_a[best_next_action_a]

        # Update the Q-value for the chosen action
        action_index = self.action_to_index(action)
        target_q_values_t = q_values_t.copy()  # Create a copy to modify
        target_q_values_a = q_values_a.copy()  # Create a copy to modify
        target_q_values_t[action_index[0]] = target_t
        target_q_values_a[action_index[1]] = target_a

        return target_q_values_t, target_q_values_a

    def prepare_q_values_for_post_episode_fitting(self, batch):
        # Calculate target Q-values for each state-action pair in the batch
        target_q_values_t = []
        target_q_values_a = []
        for state, action, reward, next_state, done in batch:
            target_q_values_t_for_state, target_q_values_a_for_state = self._calculate_target_q_value(
                state, action, reward, next_state, done)
            target_q_values_t.append(target_q_values_t_for_state)
            target_q_values_a.append(target_q_values_a_for_state)

        t = np.array(target_q_values_t)  # Target Q-values (2D array)
        a = np.array(target_q_values_a)  # Target Q-values (2D array)

        return t, a

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
