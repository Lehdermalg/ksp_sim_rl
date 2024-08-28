import matplotlib.pyplot as plt
import matplotlib
import os

from environment import SimpleKSPEnv


# Create the output folder if it doesn't exist
graphs_folder = "episode_images"

# --- Trajectory Color ---
cmap = matplotlib.colormaps['autumn']


def visualize_rocket_flight(
        episode: int,
        environment: SimpleKSPEnv,
        trajectory_x_y: list,
        scale: float = 1000.0
):
    """Simulates a rocket drop and visualizes the trajectory.

    Args:
        episode: the episode number.
        environment: The SimpleKSPEnv environment.
        scale: defaults to km scale
        trajectory_x_y: rocket flight trajectory coordinates.
    """
    from matplotlib.patches import Circle, Annulus

    # Set up the plot
    fig, ax = plt.subplots(figsize=(64, 64))
    ax.set_aspect('equal')  # Ensure circular representation
    plt.title('Rocket trajectory')

    # Kerbin representation
    kerbin_radius_km = environment.planet.radius_m / scale  # Kerbin radius in kilometers
    kerbin_circle = Circle(
        xy=(0, 0),
        radius=kerbin_radius_km,
        color='green',
        label='Kerbin'
    )
    ax.add_patch(kerbin_circle)

    # Atmosphere representation
    atmosphere_altitude_km = environment.planet.altitude_cutoff_m / scale  # Atmosphere altitude in km
    atmosphere_circle = Annulus(
        xy=(0, 0),
        r=kerbin_radius_km+atmosphere_altitude_km,
        width=atmosphere_altitude_km,
        color='blue',
        alpha=0.3,
        label='Kerbin Atmosphere'
    )
    ax.add_patch(atmosphere_circle)

    # Target altitude representation
    target_altitude_km = environment.ship.target_alt_m / scale
    # print(f'Target altitude: {target_altitude_km}')
    target_orbit = Annulus(
        xy=(0, 0),
        r=kerbin_radius_km + target_altitude_km + 2.5,
        width=5.0,  # in km
        color='red',
        alpha=0.3,
        label='Target Orbit'
    )
    ax.add_patch(target_orbit)

    # Initial rocket position
    initial_x = environment.ship.position_m[0] / scale
    initial_y = environment.ship.position_m[1] / scale
    # print(f'Initial position: {initial_x}, {initial_y}')
    ax.plot(initial_x, initial_y, 'r,', label='Initial Position')

    # Set plot limits for better visualization
    limit = kerbin_radius_km + 3.0*atmosphere_altitude_km
    ax.set_xlim(- limit, limit)
    ax.set_ylim(- limit, limit)
    # Dark violet background
    ax.set_facecolor('#100020')
    # ax.grid(True, linestyle='-')

    # --- Plotting the trajectory - one-go ---
    for step in range(len(trajectory_x_y)):
        # Calculate color based on time (step)
        color = cmap((step % 100) / 100.0)  # Normalize step to 0-1
        ax.plot(trajectory_x_y[step][0], trajectory_x_y[step][1], ',', markersize=1, color=color)

    plt.legend()
    plt.draw()

    # Save the image (overwrites previous image with the same name)
    image_trajectory_filename = os.path.join(graphs_folder, f"episode_trajectory_{episode:04d}.png")
    plt.savefig(image_trajectory_filename)

    return fig, ax


def plot_episode_data(
        episode: int,
        time: list,
        position: list,
        velocity: list,
        acceleration: list,
        step_rewards: list,
        throttle_angle: list):
    """Plots key rocket parameters over an episode.

        episode: the episode number.
        time: numpy array of time data [s]
        position: numpy array with the position data [m] (r, fi)
        velocity: numpy array with the velocity data [m/s] (r, fi)
        acceleration: numpy array with the acceleration data [m/s²] (r, fi)
        step_rewards: numpy array with the rewards
        throttle_angle: list with the throttle angle
    """

    offset_pos = 600e+3  # m
    scale_pos = 1e+0     # m
    scale_vel = 1e+0     # m/s
    scale_acc = 1e+0     # m/s²
    scale_rew = 1e+0     # 1

    fig = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    fig.suptitle(f"Rocket Parameters and Rewards - Episode #{episode}")

    # --- Create gridspec for subplot layout ---
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

    # --- Radial Component ---
    ax_radial_pos = fig.add_subplot(gs[0, 0])
    ax_radial_pos.plot(time, [(x[0] - offset_pos) * scale_pos for x in position], "b-",
                       label=f"Radial Position (m) x {scale_pos}")
    ax_radial_pos.set_ylabel("Radial")
    ax_radial_pos.set_ylim(60000, 90000)  # Set y-axis limits for radial position
    ax_radial_pos.set_yticks([_ for _ in range(60000, 95000, 5000)])  # Set y-axis ticks
    ax_radial_pos.grid(axis='y')
    ax_radial_pos.minorticks_on()
    ax_radial_pos.legend()

    ax_radial_vel = fig.add_subplot(gs[1, 0])
    ax_radial_vel.plot(time, [v[0] * scale_vel for v in velocity], "g-",
                       label=f"Radial Velocity (m/s) x {scale_vel}")
    ax_radial_vel.set_ylim(-600, 600)  # Set y-axis limits for radial velocity
    ax_radial_vel.set_yticks([_ for _ in range(-600, 800, 200)])  # Set y-axis ticks
    ax_radial_vel.grid(axis='y')
    ax_radial_vel.minorticks_on()
    ax_radial_vel.legend()

    ax_radial_acc = fig.add_subplot(gs[2, 0])
    ax_radial_acc.plot(time, [a[0] * scale_acc for a in acceleration], "r-",
                       label=f"Radial Acceleration (m/s²) x {scale_acc}")
    ax_radial_acc.set_ylim(-30, 30)  # Set y-axis limits for radial acceleration
    ax_radial_acc.set_yticks([_ for _ in range(-30, 40, 10)])  # Set y-axis ticks
    ax_radial_acc.grid(axis='y')
    ax_radial_acc.minorticks_on()
    ax_radial_acc.legend()

    # --- Angular Component ---
    ax_angular_pos = fig.add_subplot(gs[0, 1])
    ax_angular_pos.plot(time, [x[1] * scale_pos for x in position], "b-",
                        label="Angular Position (deg)")
    ax_angular_pos.set_ylabel("Angular")
    ax_angular_pos.set_ylim(0, 360)  # Set y-axis limits for angular position
    ax_angular_pos.set_yticks([_ for _ in range(0, 405, 45)])  # Set y-axis ticks
    ax_angular_pos.grid(axis='y')
    ax_angular_pos.minorticks_on()
    ax_angular_pos.legend()

    ax_angular_vel = fig.add_subplot(gs[1, 1])
    ax_angular_vel.plot(time, [v[1] * scale_vel for v in velocity], "g-",
                        label="Angular Velocity (m/s)")
    ax_angular_vel.set_ylim(-2500, 2500)  # Set y-axis limits for angular velocity
    ax_angular_vel.set_yticks([_ for _ in range(-2500, 3500, 1000)])  # Set y-axis ticks
    ax_angular_vel.grid(axis='y')
    ax_angular_vel.minorticks_on()
    ax_angular_vel.legend()

    ax_angular_acc = fig.add_subplot(gs[2, 1])
    ax_angular_acc.plot(time, [a[1] * scale_acc for a in acceleration], "r-",
                        label="Angular Acceleration (m/s²)")
    ax_angular_acc.set_ylim(-30, 30)  # Set y-axis limits for angular velocity
    ax_angular_acc.set_yticks([_ for _ in range(-30, 40, 10)])  # Set y-axis ticks
    ax_angular_acc.grid(axis='y')
    ax_angular_acc.minorticks_on()
    ax_angular_acc.legend()

    # get every n-th item of rewards data
    nth = int(len(step_rewards) / len(time))

    # --- Rewards ---
    ax_rewards = fig.add_subplot(gs[3, 0])  # Span both columns in the second row
    ax_rewards.plot(time, [step * scale_rew for step in step_rewards[::nth]], label=f"Step Reward x {scale_rew}")
    # ax_rewards.plot(time, cumulative_rewards[::nth], label="Cumulative Reward")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.set_xlabel("Time (s)")
    ax_rewards.legend()

    # --- Throttle angle ---
    ax_throttle = fig.add_subplot(gs[3, 1])
    ax_throttle.plot(time, throttle_angle, label=f"Throttle angle")
    # ax_rewards.plot(time, cumulative_rewards[::nth], label="Cumulative Reward")
    ax_throttle.set_ylabel("Throttle angle")
    ax_throttle.set_xlabel("Time (s)")
    ax_throttle.set_ylim(0, 360)  # Set y-axis limits for angular position
    ax_throttle.set_yticks([_ for _ in range(0, 405, 45)])  # Set y-axis ticks
    ax_throttle.grid(axis='y')
    ax_throttle.minorticks_on()
    ax_throttle.legend()

    # Save the image
    image_filename = os.path.join(graphs_folder, f"episode_data_{episode:04d}.png")
    plt.savefig(image_filename)
    plt.close(fig)  # Close to release resources
