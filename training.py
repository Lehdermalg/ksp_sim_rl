import matplotlib.pyplot as plt
import logging
import os
import tensorflow as tf
from datetime import datetime
import numpy as np

from environment import SimpleKSPEnv
from agents import QLearningAgentANN
from tables import reset_table, add_row
from graphs import visualize_rocket_flight, plot_episode_data

# ... (Your other constants and setup)


class RocketLearningSession:
    time = []
    rocket_trajectory_x_y = []
    rocket_position_r_fi = []
    rocket_velocity_r_fi = []
    rocket_acceleration_r_fi = []
    rocket_throttle_fi = []
    step_reward = []
    results_table = None
    episode_rewards = []
    episode_cumulative_rewards = []
    training_rewards = []

    def __init__(
            self,
            env_params: dict,
            agent_params: dict,
            training_params: dict
    ):

        self.env_params = env_params
        self.agent_params = agent_params
        self.training_params = training_params

        self.scale = self.training_params['scale']
        self.restart_episode_number = self.training_params['restart_episode_number']
        self.num_training_episodes = self.training_params['num_training_episodes']
        self.num_verification_episodes = self.training_params['num_verification_episodes']
        self.epsilon_restart = self.training_params['epsilon_restart']
        self.flights_recorded = self.training_params['flights_recorded']
        self.flight_seconds_replayed = self.training_params['flight_seconds_replayed']
        self.small_epochs = self.training_params['small_epochs']
        self.large_epochs = self.training_params['large_epochs']
        self.batch_size = self.flight_seconds_replayed
        # self.batch_size = int(self.flight_seconds_replayed /
        #                       self.env_params['step_size_s'])
        self._load_checkpoint = self.training_params['load_checkpoint']
        self._train_on_old_experience = self.training_params['train_on_old_experience']
        self._train_on_new_experience = self.training_params['train_on_new_experience']
        self.training_run = self.training_params['training_run']
        self.verification_run = self.training_params['verification_run']

        self._setup_folders(self.training_params['folder'])

        self.env = SimpleKSPEnv(**env_params)
        self.agent = QLearningAgentANN(
            env=self.env,
            writer=tf.summary.create_file_writer(self.log_folder),
            **agent_params
        )
        self.agent.small_epochs = self.small_epochs

        self._setup_checkpoint()

        self.episode_rewards = []

    def _setup_folders(self, working_folder=""):
        from graphs import graphs_folder
        from tables import tables_folder
        from agents import buffer_folder, experience_folder

        # Get the absolute path to the repository's root directory
        # Assuming your script is one level deep
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # Construct the checkpoint path relative to the root

        self.working_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), working_folder)
        self.log_folder = os.path.join(self.working_folder, 'training', 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.load_checkpoint_folder = os.path.join(self.working_folder,
                                                   'training',
                                                   'checkpoints',
                                                   self.training_params['checkpoint_folder'])
        self.save_checkpoint_folder = os.path.join(self.working_folder,
                                                   'training',
                                                   'checkpoints')

        self.graphs_folder = os.path.join(self.working_folder, graphs_folder)
        self.buffer_folder = os.path.join(self.working_folder, buffer_folder)
        self.experience_folder = os.path.join(self.working_folder, experience_folder)
        print(f"Experience folder: {self.experience_folder}")
        self.tables_folder = os.path.join(self.working_folder, tables_folder)

        # Create the output folders if they don't exist
        os.makedirs(self.graphs_folder, exist_ok=True)
        os.makedirs(self.buffer_folder, exist_ok=True)
        os.makedirs(self.tables_folder, exist_ok=True)

    def _setup_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.agent.optimizer,
            model=self.agent.q_network
        )
        self.load_checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.load_checkpoint_folder,
            max_to_keep=1
        )
        self.save_checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.save_checkpoint_folder,
            max_to_keep=5
        )

    def load_checkpoint(self):
        if self.load_checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.load_checkpoint_manager.latest_checkpoint)
            logging.info(f"Restored from {self.load_checkpoint_manager.latest_checkpoint}")
        else:
            logging.warning(f"NOT RESTORED CHECKPOINT")
            logging.warning(f"Checkpoint manager LOAD directory: {self.load_checkpoint_folder}")
            logging.warning(f"Checkpoint manager SAVE directory: {self.save_checkpoint_folder}")
            logging.warning(f"Checkpoint manager last checkpoint: {self.load_checkpoint_manager.latest_checkpoint}")

    def pretrain_agent(self):
        self.agent.load_experience_from_folder(replay_folder=self.experience_folder)
        self.agent.train_on_experience(
            large_epochs=self.large_epochs,
            small_epochs=self.small_epochs,
            batch_size=self.batch_size
        )

    def run_step(self, state, action, episode_num, verify: bool = False):
        loss = None
        step_s = round(self.env.t / self.env.dt)

        # --- Environment action step
        next_state, reward, done, info = self.env.step(action)
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) == 1:
            self.episode_cumulative_rewards.append(reward)
        else:
            self.episode_cumulative_rewards.append(self.episode_cumulative_rewards[-1] + reward)

        if not verify and round(self.env.t, 3) % 1 == 0:
            loss = self.agent.update(state, action, reward, next_state, done, step_s)

        state = next_state

        # Store experience in the replay buffer
        self.agent.flight_replay_buffer.add((state, action, reward, next_state, done))

        # logging.info(f"Step: {step_s} Loss: {loss} Action: {action}")
        # logging.info(f"Action: {action}\nState: {state}")

        if round(self.env.t, 3) % 1 == 0:
            # --- Log data for TensorBoard ---
            with self.agent.writer.as_default():
                tf.summary.scalar('Episode Reward', self.episode_rewards[-1], step=step_s)
                tf.summary.scalar('Episode Cumulative Reward', self.episode_cumulative_rewards[-1], step=step_s)
                if not verify:
                    tf.summary.scalar('Loss', loss, step=step_s)  # Assuming you have a 'loss' variable
                tf.summary.scalar('Epsilon', self.agent.epsilon, step=step_s)
                tf.summary.scalar('Throttle', action[0], step=step_s)
                tf.summary.scalar('Thrust Angle', action[1], step=step_s)
                # tf.summary.histogram('Q-Values', self.agent.q_values[:, 0], step=step_s)  # Log Q-values

            # --- Action policy
            action = self.agent.choose_action(state)

            step_l = int(self.env.t)
            ship_r = np.linalg.norm(self.env.ship.position_m)
            # Give the output some love
            logging.info(
                f"Time: {step_l:3.2f}, "
                f"altitude: {self.env.ship.current_alt_m:7.0f}, "
                f"rewards: {self.episode_cumulative_rewards[-1]:6.2f}, "
                f"throttle: {self.env.ship.throttle:4.1f}, "
                f"angle: {self.env.ship.thrust_angle:5.2f}, "
                f"fuel: {self.env.ship.total_fuel_mass_kg:8.1f} "
                # f"LOSS: {self.env.loss} "
                f"velocity R [m/s]: {self.env.ship.velocity_r_fi_mps[0]:8.1f} "
                f"velocity φ [m/s]: {self.env.ship.velocity_r_fi_mps[1]:8.1f} "
                f"loss: {loss}"
            )
            # Be kind to your final tables
            add_row(self.results_table, self.env, self.episode_cumulative_rewards[-1])
            # And don't forget to plot
            self.rocket_trajectory_x_y.append(self.env.ship.position_m / self.scale)

            self.time.append(round(self.env.t, 3))
            self.rocket_position_r_fi.append(self.env.ship.position_r_fi_m)
            self.rocket_velocity_r_fi.append(self.env.ship.velocity_r_fi_mps)
            self.rocket_acceleration_r_fi.append(self.env.ship.acceleration_r_fi_mps2)
            self.rocket_throttle_fi.append(self.env.ship.thrust_angle)
            # print(f"DEBUG: {rocket_trajectory_x}, {rocket_trajectory_y}")
            # --- Generate and save the plot ---
            fig, ax = visualize_rocket_flight(
                episode=episode_num,
                environment=self.env,
                scale=self.scale,
                trajectory_x_y=self.rocket_trajectory_x_y,
                folder=self.graphs_folder
            )

            # --- Plotting the trajectory data ---
            plot_episode_data(
                episode=episode_num,
                time=self.time,
                position=self.rocket_position_r_fi,
                velocity=self.rocket_velocity_r_fi,
                acceleration=self.rocket_acceleration_r_fi,
                step_rewards=self.episode_rewards,
                throttle_angle=self.rocket_throttle_fi,
                folder=self.graphs_folder
            )

            # Save the image (overwrites previous image with the same name)
            plt.close(fig)  # Close the figure to release resources

        if done:
            logging.info(f".. STOPPING ..")
            logging.info(f"Final episode reward BEFORE crash punishment: {self.episode_cumulative_rewards[-1]}")
            logging.info(f"Episode crash punishment: {self.env.crash_punishment}")
            if self.env.ship.crashed:
                if self.env.crash_punishment is None:
                    self.env.crash_punishment = -0.99 * self.episode_cumulative_rewards[-1]
                # FIRST Prepare the crash penalty for a new run (cumulative rewards WITHOUT the crash penalty applied)
                _new_crash_punishment = -1.01 * self.episode_cumulative_rewards[-1]

                # THEN Apply the crash penalty
                self.step_reward += self.env.crash_punishment
                self.episode_rewards[-1] += self.env.crash_punishment
                self.episode_cumulative_rewards[-1] += self.env.crash_punishment
                logging.info(f"Modifying crash punishment: {self.env.crash_punishment}")

                # FINALLY Store the crash penalty
                self.env.crash_punishment = _new_crash_punishment

            logging.info(f"Final episode reward AFTER crash punishment: {self.episode_cumulative_rewards[-1]}")

            if not verify:
                self.agent.update(state, action, reward, next_state, done, step_s)

            # --- Save the FlightBuffer to a file ---
            buffer_filename = os.path.join(self.buffer_folder, f"episode_{episode_num:04d}_flight_buffer.json")

            self.agent.flight_replay_buffer.final_reward = self.episode_cumulative_rewards[-1]
            self.training_rewards.append(self.episode_cumulative_rewards[-1])
            if self.agent.flight_replay_buffer.final_reward > 0:
                self.agent.flight_replay_buffer.save(buffer_filename)
                logging.info(f"Adding new flight to buffer "
                             f"- size {len(self.agent.multi_replay_buffer.flight_list)} "
                             f"- reward: {self.agent.flight_replay_buffer.final_reward}")
                self.agent.multi_replay_buffer.add(self.agent.flight_replay_buffer)

        return state, action

    def reset(self):
        logging.info(f"Resetting the training")
        logging.info(f"Ship position: {self.env.ship.position_m}")
        # --- Randomize the initial altitude and velocity ---
        self.env.randomize_position()
        logging.info(f"Ship position: {self.env.ship.position_m}")
        self.env.randomize_velocity()
        logging.info(f"Ship position: {self.env.ship.position_m}")

        self.env.reset(
            start_position_m=self.env.ship.position_m,
            start_velocity_mps=self.env.ship.velocity_mps,
        )
        logging.info(f"Ship position: {self.env.ship.position_m}")

        # prepare the tables
        self.results_table = reset_table()
        # and graphs
        self.time = []
        self.episode_rewards = []
        self.rocket_trajectory_x_y = []
        self.rocket_position_r_fi = []
        self.rocket_velocity_r_fi = []
        self.rocket_acceleration_r_fi = []
        self.rocket_throttle_fi = []
        self.step_reward = []

    def run_episode(self, episode_num: int, verify: bool = False):
        # --- Batch learning ---
        if episode_num % self.epsilon_restart == 0:
            self.agent.epsilon = self.agent.max_epsilon

        self.reset()
        # Initial state and action
        state = self.env.get_observation()
        action = np.array([0.0, 0.0])

        while not self.env.done:
            state, action = self.run_step(state, action, episode_num, verify=verify)

        if len(self.rocket_trajectory_x_y) > 0:
            # --- Generate and save the plot ---
            fig, ax = visualize_rocket_flight(
                episode=episode_num,
                environment=self.env,
                scale=self.scale,
                trajectory_x_y=self.rocket_trajectory_x_y,
                folder=self.graphs_folder
            )
            ax.plot(self.rocket_trajectory_x_y[:][0],
                    self.rocket_trajectory_x_y[:][1],
                    'r,',
                    markersize=1,
                    # label='Rocket Trajectory'
                    )
            plt.show()

        # --- Save the PrettyTable to a file ---
        table_filename = os.path.join(self.tables_folder, f"episode_{episode_num:04d}_table.txt")
        with open(table_filename, "w", encoding="utf-8") as f:
            f.write(str(self.results_table))  # Write the table string to the file

        logging.info(self.results_table)
        logging.info(f'Episode #{episode_num} reward: {self.training_rewards[-1]}')

        # Sample a batch from the multi-flight replay buffer
        if not verify:
            if self._train_on_new_experience:
                self.train_on_buffer()
            save_path = self.save_checkpoint_manager.save()
            logging.info(f'Saving episode {episode_num} checkpoint at {save_path}')
        else:
            buffer_filename = os.path.join(self.buffer_folder, f"episode_{episode_num:04d}_flight_buffer.json")
            self.agent.flight_replay_buffer.save(buffer_filename)

    def train_on_buffer(self):
        # Sample a batch from the multi-flight replay buffer
        batch = self.agent.multi_replay_buffer.sample(
            batch_size=self.flight_seconds_replayed,
            n_th=int(1/self.env.dt)
        )
        if len(batch) > 0:
            states, actions, rewards, next_states, dones = zip(*batch)

            # Prepare data for training
            x = np.array(states)
            # Convert actions to indices (assuming you have a way to map actions to indices)
            # y = self.agent.prepare_q_values_for_post_episode_fitting(batch)
            y = self.agent.prepare_target_action_indices(batch)

            # Perform a batch update using model.fit
            self.agent.q_network.fit(x, y, epochs=self.small_epochs, verbose=0)  # Single epoch, no output

    def train(
            self,
            load: bool = True,
            train_on_old_experience: bool = True,
            train_on_new_experience: bool = True
    ):
        if load:
            self.load_checkpoint()  # Load checkpoint if available
        if train_on_old_experience:
            self.pretrain_agent()   # Pretrain if needed

        for episode in range(self.restart_episode_number,
                             self.restart_episode_number + self.num_training_episodes):
            self.run_episode(episode_num=episode)

            # Sample and train on experience after each episode (if applicable)
            if train_on_new_experience:
                self.train_on_buffer()

    def verify(self, epsilon: float):
        logging.info(f"Loading checkpoint")
        self.load_checkpoint()  # Load the trained model
        self.agent.max_epsilon = epsilon
        self.agent.min_epsilon = epsilon
        self.agent.epsilon_decay = 0.0

        # Run verification episodes (similar to run_episode, but without training)
        for episode in range(0, self.num_verification_episodes):
            self.run_episode(episode_num=episode, verify=True)  # Or adjust as needed

        for episode in range(0, self.num_verification_episodes):
            logging.info(f"Verification reward: {self.training_rewards[episode]}")


# --- Main Script ---
if __name__ == "__main__":
    # Define parameters for environment, agent, and training
    env_params = {
        'total_time_s': 2000,
        'step_size_s': 1.0e-2,
        'target_altitude_m': 75.0e+3,
        'start_altitude_m_lo': 60.5e+3,
        'start_altitude_m_hi': 61.5e+3,
        'start_angle_deg_lo': 0.0,
        'start_angle_deg_hi': 0.0,
        'start_velocity_percent_orbital_lo': 0.0,
        'start_velocity_percent_orbital_hi': 0.0,
        'altitude_cutoff_lo': 60.0e+3,
        'altitude_cutoff_hi': 90.0e+3,
        # ... other environment parameters
    }
    agent_params = {
        'learning_rate': 1.0e-3,
        'gamma': 1.0 - 5e-1,
        'epsilon_lo': 1.0e-2,
        'epsilon_hi': 75.0e-2,
        'epsilon_decay': 1.0-1e-3,
        'flights_recorded': 5,
        'flight_steps_recorded': 100*100  # seconds * steps/s
        # TODO: needs improving to just use seconds
    }
    training_params = {
        'restart_episode_number': 0,
        'num_training_episodes': 10,
        'num_verification_episodes': 15,  # Add this for verification runs
        'epsilon_restart': 5,
        'flights_recorded': 5,
        'flight_seconds_replayed': 50,
        'large_epochs': 1,
        'small_epochs': 1,
        'scale': 1e+3,
        'folder': 'verification'
    }

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create the learning session
    session = RocketLearningSession(env_params, agent_params, training_params)

    # Trigger learning or verification
    # session.train()
    session.verify(epsilon=0.0)  # Uncomment to run verification episodes
