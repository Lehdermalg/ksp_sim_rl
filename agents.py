import numpy as np
import tensorflow as tf
import logging


class CustomCallback(tf.keras.callbacks.Callback):
    @staticmethod
    def on_batch_end(batch, rewards):
        if batch % 10 == 0:
            logging.info(f"Batch {batch}: Rewards = {rewards[-1]:.4f}")


class QLearningAgentANN(object):
    def __init__(
            self,
            env,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Build the neural network (you can customize the architecture)
        self.q_network = self._build_q_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_q_network(self):
        """Creates the neural network for Q-value approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.env.observation_space.shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            # Output: Q-values for each action
            # Sigmoid to correctly map to the action space
            tf.keras.layers.Dense(self.env.action_space.shape[0], activation='sigmoid')
        ])
        logging.info(f"Action space shape: {self.env.action_space.shape[0]}")
        return model

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_network.predict(state[np.newaxis, :])[0]
            throttle = (self.env.action_space.low[0] + q_values[0] *
                        (self.env.action_space.high[0] - self.env.action_space.low[0]))
            thrust_angle = (self.env.action_space.low[1] + q_values[1] *
                            (self.env.action_space.high[1] - self.env.action_space.low[1]))
            return np.array([throttle, thrust_angle])

    def update(self, state, action, reward, next_state, done, step):
        """Updates the Q-network using a gradient descent step."""
        with tf.GradientTape() as tape:
            q_values = self.q_network(state[np.newaxis, :])
            q_value = tf.reduce_sum(q_values * tf.one_hot(np.argmax(action), self.env.action_space.shape[0]), axis=1)
            print(f"Q value: {q_value}")

            if done and next_state.size == 0:   # Handle the crash
                target = reward
            else:
                next_q_values = self.q_network(next_state[np.newaxis, :])
                print(f"Next Q values: {next_q_values}")
                max_next_q_value = tf.reduce_max(next_q_values, axis=1)
                print(f"Max next Q value: {max_next_q_value}")
                target = reward + self.gamma * max_next_q_value
                print(f"Target: {target}")

            loss = tf.keras.losses.MeanSquaredError()(target, q_value)

            # Log Q-values
            tf.summary.histogram('Q-Values', q_values, step=step)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        # print(f"Grads: {grads}")
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss
