import matplotlib.pyplot as plt
import logging
import os
import tensorflow as tf
from datetime import datetime
import numpy as np

from environment import SimpleKSPEnv
from agents import QLearningAgentANN
from tables import reset_table, add_row
from graphs import visualize_rocket_flight, plot_episode_data, graphs_folder
from maths import normalize, rotate_vector_by_angle

current_folder = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = "training/checkpoints"  # File name
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint_path = os.path.join(current_folder, os.path.dirname(checkpoint_dir))

log_dir = "training/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Main Script ---
if __name__ == "__main__":
    # Create the environment
    _t = 2000       # simulation time limit [s]
    _ht = 75.0e+3   # desired orbital altitude
    _h0_rel = np.array([0.0e+0, 75.0e+3])    # initial altitude - relative to planet surface
    _v = 1.0        # at this % of orbital velocity the Rp = ground level

    ske = SimpleKSPEnv(total_time_s=_t)
    # Position magic
    ske.set_target_altitude_m(_ht)
    _h0_abs = (np.linalg.norm(_h0_rel) + ske.planet.radius_m) * normalize(_h0_rel)  # absolute initial position
    ske.set_initial_position_m(_h0_abs)
    # Velocity magic
    orbital_velocity = ske.planet.orbital_velocity_mps(altitude_m=ske.ship.target_alt_m)
    _v0 = rotate_vector_by_angle(normalize(_h0_rel), 90) * _v * orbital_velocity
    ske.set_initial_velocity_mps(_v0)

    # Create the graphs output folder if it doesn't exist
    os.makedirs(graphs_folder, exist_ok=True)

    # Initialize the agent
    agent = QLearningAgentANN(
        env=ske,
        learning_rate=0.01,
        gamma=1.0-5e-5,            # Discount factor - high for long-term rewards
        # 1.0-5e-4 ==>  2.000 steps into the past =>  20.0 s
        # 1.0-5e-5 ==> 20.000 steps into the past => 200.0 s
        epsilon=0.9,               # High exploration to start with
        epsilon_decay=1.0-1e-4,    # To be adjusted
        min_epsilon=0.01           # Minimum exploration
    )

    # Set up the checkpoint and the manager
    checkpoint = tf.train.Checkpoint(
        optimizer=agent.optimizer,
        model=agent.q_network
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_path,
        max_to_keep=5
    )
    # Restart from checkpoint if possible
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        logging.info(f"Restored from {checkpoint_manager.latest_checkpoint}")

    # --- Training Loop ---
    writer = tf.summary.create_file_writer(log_dir)  # Create a SummaryWriter

    # Trial to start this journey
    restart_episode_number = 0
    num_episodes = 1
    final_episode_number = num_episodes + restart_episode_number
    episode_rewards = []
    for episode in range(restart_episode_number, final_episode_number):
        # --- Randomize the initial altitude ---
        altitude_variation = (np.random.rand() - 0.5) * 1000.0  # Random variation between -500m and +500m
        initial_altitude = (np.linalg.norm(_h0_rel) + ske.planet.radius_m + altitude_variation) * normalize(_h0_rel)  # absolute initial position
        velocity_variation = 1.0
        # Random variation between -value..+value
        # velocity_randomization = np.random.rand() * 2.0 - 1.0
        initial_velocity = _v * orbital_velocity * velocity_variation * rotate_vector_by_angle(normalize(_h0_rel), 90)

        state = ske.reset(
            position_m=initial_altitude,
            velocity_mps=initial_velocity,
        )

        # prepare the tables
        results_table = reset_table()
        # and graphs
        scale = 1000  # for km scale
        time = []
        rocket_trajectory_x_y = []
        rocket_position_r_fi = []
        rocket_velocity_r_fi = []
        rocket_acceleration_r_fi = []
        step_reward = []

        while not ske.done:
            # action = agent.choose_action(state)
            action = np.array([0.0, 0.0])
            next_state, reward, done, info = ske.step(action)
            # loss = agent.update(state, action, reward, next_state, done)

            state = next_state
            step_s = round(ske.t/ske.dt)

            # logging.info(f"Step: {step_s} Loss: {loss} Action: {action}")
            # logging.info(f"Action: {action}\nState: {state}")

            # --- Log data for TensorBoard ---
            if False:
                with writer.as_default():
                    tf.summary.scalar('Episode Reward', ske.episode_rewards[-1], step=step_s)
                    tf.summary.scalar('Loss', loss.numpy(), step=step_s)  # Assuming you have a 'loss' variable
                    tf.summary.scalar('Epsilon', agent.epsilon, step=step_s)

            if round(ske.t, 3) % 1 == 0:
                step_l = int(ske.t)
                ship_r = np.linalg.norm(ske.ship.position_m)
                # Give the output some love
                logging.info(
                    f"Time: {step_l:3.2f}, "
                    f"altitude: {ske.ship.current_alt_m:7.0f}, "
                    # f"rewards: {ske.episode_rewards:6.2f}, "
                    f"throttle: {ske.ship.throttle:4.1f}, "
                    f"angle: {ske.ship.thrust_angle:5.2f}, "
                    f"fuel: {ske.ship.total_fuel_mass_kg:8.1f} "
                    # f"LOSS: {ske.loss} "
                    f"velocity R [m/s]: {ske.ship.velocity_r_fi_mps[0]:8.1f} "
                    f"velocity φ [m/s]: {ske.ship.velocity_r_fi_mps[1]/(2*np.pi) * ship_r:8.1f} "
                )
                # Be kind to your final tables
                add_row(results_table, ske)
                # And don't forget to plot
                rocket_trajectory_x_y.append(ske.ship.position_m / scale)

                time.append(round(ske.t, 3))
                rocket_position_r_fi.append(ske.ship.position_r_fi_m)
                rocket_velocity_r_fi.append(ske.ship.velocity_r_fi_mps)
                rocket_acceleration_r_fi.append(ske.ship.acceleration_r_fi_mps2)
                # print(f"DEBUG: {rocket_trajectory_x}, {rocket_trajectory_y}")
                # --- Generate and save the plot ---
                fig, ax = visualize_rocket_flight(
                    episode=episode,
                    environment=ske,
                    scale=scale,
                    trajectory_x_y=rocket_trajectory_x_y
                )

                # --- Plotting the trajectory data ---
                plot_episode_data(
                    episode=episode,
                    time=time,
                    position=rocket_position_r_fi,
                    velocity=rocket_velocity_r_fi,
                    acceleration=rocket_acceleration_r_fi,
                    step_rewards=ske.episode_rewards,
                    cumulative_rewards=ske.cumulative_rewards)

                # Save the image (overwrites previous image with the same name)
                plt.close(fig)  # Close the figure to release resources

            if done:
                logging.info(f".. STOPPING ..")
                break

        # --- Generate and save the plot ---
        fig, ax = visualize_rocket_flight(
            episode=episode,
            environment=ske,
            scale=scale,
            trajectory_x_y=rocket_trajectory_x_y
        )
        ax.plot(rocket_trajectory_x_y[:][0],
                rocket_trajectory_x_y[:][1],
                'r,',
                markersize=1,
                # label='Rocket Trajectory'
                )
        plt.show()

        # --- Save the PrettyTable to a file ---
        output_folder = "episode_tables"
        os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

        table_filename = os.path.join(output_folder, f"episode_{episode:04d}_table.txt")
        with open(table_filename, "w", encoding="utf-8") as f:
            f.write(str(results_table))  # Write the table string to the file

        logging.info(results_table)
        total_reward = sum(ske.episode_rewards)
        episode_rewards.append(total_reward)
        logging.info(f'Episode #{episode} reward: {total_reward}')

        save_path = checkpoint_manager.save()
        logging.info(f'Saving episode {episode} checkpoint at {save_path}')

    plt.show()  # It will block further plot execution!
