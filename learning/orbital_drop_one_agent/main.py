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
    # _position_angle = np.random.rand() * 360.0
    _position_angle = np.random.rand() * 5.0
    # initial altitude - relative to planet surface
    _h0_rel = rotate_vector_by_angle(np.array([0.0e+0, 75.0e+3]), _position_angle)
    _v = 0.1        # at 60% of orbital velocity the ship needs approx 6s thrust to get to 100%
    _hard_coded_policy_test = False

    ske = SimpleKSPEnv(total_time_s=_t)
    # Position magic
    ske.set_target_altitude_m(_ht)
    _h0_abs = (np.linalg.norm(_h0_rel) + ske.planet.radius_m) * normalize(_h0_rel)  # absolute initial position
    ske.set_initial_position_m(_h0_abs)
    # Velocity magic
    orbital_velocity_mps = ske.planet.orbital_velocity_mps(altitude_m=ske.ship.target_alt_m)
    _v0 = rotate_vector_by_angle(normalize(_h0_rel), 90) * _v * orbital_velocity_mps
    ske.set_initial_velocity_mps(_v0)

    # Create the graphs output folder if it doesn't exist
    os.makedirs(graphs_folder, exist_ok=True)

    # --- Training Loop ---
    writer = tf.summary.create_file_writer(log_dir)  # Create a SummaryWriter

    # Initialize the agent
    _epsilon_start = 0.7
    agent = QLearningAgentANN(
        env=ske,
        learning_rate=0.01,
        gamma=1.0-5e-2,            # Discount factor - high for long-term rewards
        # 1.0-5e-1 ==>       2 steps into the past =>    0.02  s
        # 1.0-5e-2 ==>      20 steps into the past =>    0.20  s
        # 1.0-5e-3 ==>     200 steps into the past =>    2.00  s
        # 1.0-5e-4 ==>   2.000 steps into the past =>   20.00  s
        # 1.0-5e-5 ==>  20.000 steps into the past =>  200.00  s
        # 1.0-5e-6 ==> 200.000 steps into the past => 2000.00  s
        epsilon=_epsilon_start,    # High exploration to start with
        epsilon_decay=1.0-1e-2,    # To be adjusted
        min_epsilon=1e-3,           # Minimum exploration
        writer=writer
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

    # Trial to start this journey
    restart_episode_number = 18
    num_episodes = 102
    epsilon_restart = 3
    final_episode_number = num_episodes + restart_episode_number
    episode_rewards = []

    action = np.array([0.0, 0.0])

    for episode in range(restart_episode_number, final_episode_number):
        # --- Batch learning ---
        if episode % epsilon_restart == 0:
            agent.epsilon = _epsilon_start

        # --- Randomize the initial altitude ---
        altitude_variation = (np.random.rand() - 0.5) * 0.5e+3  # Random variation between -0.250m and +0.250m
        # altitude_variation = (np.random.rand() - 0.5) * 1.0e+3  # Random variation between -0.500m and +0.500m
        # altitude_variation = (np.random.rand() - 0.5) * 2.0e+3  # Random variation between -1.000m and +1.000m
        # altitude_variation = (np.random.rand() - 0.5) * 12.5e+3  # Random variation between -12.500m and +12.500m
        # altitude_variation = 141  # add 141m that the rocket will fall within the approx. 6s of thrust
        # absolute initial position
        # _position_angle = np.random.rand() * 360.0
        _position_angle = np.random.rand() * 5.0  # maybe the 360 deg randomization is too much?
        initial_position_m = (np.linalg.norm(_h0_rel) + ske.planet.radius_m + altitude_variation) * normalize(_h0_rel)
        initial_position_m = rotate_vector_by_angle(initial_position_m, _position_angle)
        # initial altitude - relative to planet surface

        # velocity_variation = 1.0
        # Random variation between -0.1 v .. +0.1 v
        velocity_variation = ((np.random.rand() * 2.0 - 1.0) * 0.1 + 1.0) * _v
        initial_velocity_mps = (orbital_velocity_mps * velocity_variation *
                                rotate_vector_by_angle(normalize(initial_position_m), 90))

        state = ske.reset(
            position_m=initial_position_m,
            velocity_mps=initial_velocity_mps,
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
        rocket_throttle_fi = []
        step_reward = []

        while not ske.done:
            step_s = round(ske.t/ske.dt)

            # --- Environment action step
            next_state, reward, done, info = ske.step(action)
            if not _hard_coded_policy_test:
                if round(ske.t, 3) % 1 == 0:
                    loss = agent.update(state, action, reward, next_state, done, step_s)

            state = next_state

            # logging.info(f"Step: {step_s} Loss: {loss} Action: {action}")
            # logging.info(f"Action: {action}\nState: {state}")

            # --- Log data for TensorBoard ---
            if not _hard_coded_policy_test:
                if round(ske.t, 3) % 1 == 0:
                    with writer.as_default():
                        tf.summary.scalar('Episode Reward', ske.episode_rewards[-1], step=step_s)
                        tf.summary.scalar('Loss', loss.numpy(), step=step_s)  # Assuming you have a 'loss' variable
                        tf.summary.scalar('Epsilon', agent.epsilon, step=step_s)
                        tf.summary.scalar('Throttle', action[0], step=step_s)
                        tf.summary.scalar('Thrust Angle', action[1], step=step_s)
                        tf.summary.histogram('Q-Values Throttle', agent.q_values[:, 0], step=step_s)  # Log Q-values
                        tf.summary.histogram('Q-Values Angle', agent.q_values[:, 1], step=step_s)  # Log Q-values

            if round(ske.t, 3) % 1 == 0:

                # --- Action policy
                if _hard_coded_policy_test:
                    print(f"Current state: {state}")
                    action = np.array([0.0, 0.0])
                    if np.linalg.norm(ske.ship.velocity_mps) < orbital_velocity_mps:
                        # action = np.array([100.0, 0.0 * np.pi])  # This produced UPWARD thrust
                        action = np.array([100.0, 0.5 * np.pi])  # This produced LEFTWARD thrust
                        # action = np.array([100.0, 1.0 * np.pi])  # This produced DOWNWARD thrust
                        # action = np.array([100.0, 1.5 * np.pi])  # This produced RIGHTWARD thrust
                else:
                    action = agent.choose_action(state)

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
                    f"velocity φ [m/s]: {ske.ship.velocity_r_fi_mps[1]:8.1f} "
                )
                # Be kind to your final tables
                add_row(results_table, ske)
                # And don't forget to plot
                rocket_trajectory_x_y.append(ske.ship.position_m / scale)

                time.append(round(ske.t, 3))
                rocket_position_r_fi.append(ske.ship.position_r_fi_m)
                rocket_velocity_r_fi.append(ske.ship.velocity_r_fi_mps)
                rocket_acceleration_r_fi.append(ske.ship.acceleration_r_fi_mps2)
                rocket_throttle_fi.append(ske.ship.thrust_angle)
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
                    throttle_angle=rocket_throttle_fi)

                # Save the image (overwrites previous image with the same name)
                plt.close(fig)  # Close the figure to release resources

            if done:
                logging.info(f".. STOPPING ..")
                logging.info(f"Episode crash punishment: {ske.crash_punishment}")
                loss = agent.update(state, action, reward, next_state, done, step_s)
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
