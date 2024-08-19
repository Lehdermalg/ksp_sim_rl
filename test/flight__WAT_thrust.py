import numpy as np

from environment import SimpleKSPEnv
from tables import reset_table, add_row


def visualize_rocket_trajectory(environment, num_steps=4000):
    """Simulates a rocket hover and visualizes the trajectory.

    Args:
        environment: The SimpleKSPEnv environment.
        num_steps: The number of simulation steps to run.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Annulus

    scale = 1000
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')  # Ensure circular representation
    plt.title('Rocket Drop from Orbit')

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
    print(f'Target altitude: {target_altitude_km}')
    target_orbit = Annulus(
        xy=(0, 0),
        r=kerbin_radius_km + target_altitude_km + 5.0,
        width=5.0,  # in km
        color='red',
        alpha=0.3,
        label='Target Orbit'
    )
    ax.add_patch(target_orbit)

    # Initial rocket position
    initial_x = environment.ship.position_m[0] / scale
    initial_y = environment.ship.position_m[1] / scale
    print(f'Initial position: {initial_x}, {initial_y}')
    ax.plot(initial_x, initial_y, 'r,', label='Initial Position')

    rocket_trajectory_x = []
    rocket_trajectory_y = []

    results_table = reset_table()

    # Simulation loop
    for step in range(num_steps):
        action = np.array([100000.0, 0.0])  # WAT thrust - should be capped to 0.0-100.0
        observation, reward, done, info = environment.step(action)

        # Plot rocket position every 100 steps (every second)
        if step % 100 == 0:
            table_i = int(step/100)
            # Give the output some love
            print(f"Time: {table_i:3.0f}, "
                  f"altitude: {environment.ship.current_alt_m:7.0f}, "
                  # f"rewards: {environment.episode_rewards:6.2f}, "
                  f"throttle: {environment.ship.throttle:4.1f}, "
                  f"angle: {environment.ship.thrust_angle:5.2f}, "
                  f"fuel: {environment.ship.total_fuel_mass_kg:8.1f} "
                  # f"LOSS: {environment.loss} "
                  f"velocity R: {environment.ship.velocity_r_fi_mps[0]:8.1f} "
                  f"velocity φ: {environment.ship.velocity_r_fi_mps[1]:8.1f} "
                  )
            # Be kind to your final tables
            add_row(results_table, environment)
            # And don't forget to plot
            rocket_trajectory_x.append(environment.ship.position_m[0] / scale)
            rocket_trajectory_y.append(environment.ship.position_m[1] / scale)
            # print(f"DEBUG: {rocket_trajectory_x}, {rocket_trajectory_y}")
            ax.plot(rocket_trajectory_x[table_i],
                    rocket_trajectory_y[table_i],
                    'r,',
                    markersize=1,
                    # label='Rocket Trajectory'
                    )
            # plt.pause(0.01)
        if done:
            break

    print(results_table)

    # Set plot limits for better visualization
    limit = kerbin_radius_km + 3.0*atmosphere_altitude_km
    ax.set_xlim(- limit, limit)
    ax.set_ylim(- limit, limit)
    # Dark violet background
    ax.set_facecolor('#100020')
    # ax.grid(True, linestyle='-')
    plt.legend()
    plt.show()


# --- Main Script ---
if __name__ == "__main__":
    # Create the environment
    _t = 1000       # simulation time limit [s]
    _h = 10.0       # starting at 10m
    _v = 0.970143   # at this % of orbital velocity the Rp = ground level

    ske = SimpleKSPEnv(total_time_s=_t)
    ske.set_target_altitude_m(75.0e+3)
    ske.set_initial_position_m(np.array([_h+ske.planet.radius_m, 0.0]))
    orbital_velocity = ske.planet.orbital_velocity_mps(altitude_m=ske.ship.target_alt_m)
    ske.set_initial_velocity_mps(np.array([0.0, 0.0]))

    # Visualize the drop
    visualize_rocket_trajectory(environment=ske, num_steps=int(_t*1.0e+2))
