import matplotlib
import matplotlib.pyplot as plt
import logging
import os
import tensorflow as tf
from datetime import datetime

from environment import SimpleKSPEnv
from agents import QLearningAgentANN
from tables import reset_table, add_row
from graphs import visualize_rocket_flight


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
    _h0 = 5.0e+1    # initial altitude
    _v = 0.0        # at this % of orbital velocity the Rp = ground level

    ske = SimpleKSPEnv(total_time_s=_t)
    ske.set_target_altitude_m(_ht)
    ske.set_initial_position_m([_h0+ske.planet.radius_m, 0.0])
    orbital_velocity = ske.planet.orbital_velocity_mps(altitude_m=ske.ship.target_alt_m)
    ske.set_initial_velocity_mps([0.0, _v * orbital_velocity])

    agent = QLearningAgentANN(
        env=ske,
        learning_rate=0.1,
        gamma=1.0-1e-4,            # Discount factor - high for long-term rewards
        epsilon=0.3,            # High exploration to start with
        epsilon_decay=1.0-1e-7,    # To be adjusted
        min_epsilon=0.01        # Minimum exploration
    )

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

    # --- Trajectory Color ---
    cmap = matplotlib.colormaps['autumn']

    # Trial to start this journey
    restart_episode_number = 7
    num_episodes = 14
    final_episode_number = num_episodes + restart_episode_number
    episode_rewards = []
    for episode in range(restart_episode_number, final_episode_number):
        state = ske.reset(
            position_m=[_h0+ske.planet.radius_m, 0.0],
            velocity_mps=[0.0, _v * orbital_velocity],
        )

        # prepare the tables
        results_table = reset_table()
        # and graphs
        scale = 1000  # for km scale
        rocket_trajectory_x, rocket_trajectory_y = [], []

        while not ske.done:
            action = agent.choose_action(state)
            next_state, reward, done, info = ske.step(action)
            loss = agent.update(state, action, reward, next_state, done, )

            state = next_state
            step_s = round(ske.t/ske.dt)

            # logging.info(f"Step: {step_s} Loss: {loss} Action: {action}")
            # logging.info(f"Action: {action}\nState: {state}")

            # --- Log data for TensorBoard ---
            with writer.as_default():
                tf.summary.scalar('Episode Reward', ske.episode_rewards[-1], step=step_s)
                tf.summary.scalar('Loss', loss.numpy(), step=step_s)  # Assuming you have a 'loss' variable
                tf.summary.scalar('Epsilon', agent.epsilon, step=step_s)

            if round(ske.t, 3) % 1 == 0:
                step_l = int(ske.t)
                # Give the output some love
                logging.info(
                    f"Time: {step_l:3.2f}, "
                    f"altitude: {ske.ship.current_alt_m:7.0f}, "
                    # f"rewards: {ske.episode_rewards:6.2f}, "
                    f"throttle: {ske.ship.throttle:4.1f}, "
                    f"angle: {ske.ship.thrust_angle:5.2f}, "
                    f"fuel: {ske.ship.total_fuel_mass_kg:8.1f} "
                    # f"LOSS: {ske.loss} "
                    f"velocity R: {ske.ship.velocity_r_fi_mps[0]:8.1f} "
                    f"velocity φ: {ske.ship.velocity_r_fi_mps[1]:8.1f} "
                )
                # Be kind to your final tables
                add_row(results_table, ske)
                # And don't forget to plot
                rocket_trajectory_x.append(ske.ship.position_m[0] / scale)
                rocket_trajectory_y.append(ske.ship.position_m[1] / scale)
                # print(f"DEBUG: {rocket_trajectory_x}, {rocket_trajectory_y}")
                # --- Generate and save the plot ---
                fig, ax = visualize_rocket_flight(environment=ske, scale=scale)

                # --- Plotting the trajectory - one-go ---
                for step in range(len(rocket_trajectory_x)):
                    # Calculate color based on time (step)
                    color = cmap((step % 100) / 100.0)  # Normalize step to 0-1
                    ax.plot(rocket_trajectory_x[step], rocket_trajectory_y[step], ',', markersize=1, color=color)

                # Create the output folder if it doesn't exist
                output_folder = "episode_images"
                os.makedirs(output_folder, exist_ok=True)

                # Save the image (overwrites previous image with the same name)
                image_filename = os.path.join(output_folder, f"episode_{episode:04d}.png")
                plt.savefig(image_filename)
                plt.close(fig)  # Close the figure to release resources

            if done:
                logging.info(f".. STOPPING ..")
                break

        fig, ax = visualize_rocket_flight(environment=ske, scale=scale)
        ax.plot(rocket_trajectory_x,
                rocket_trajectory_y,
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
