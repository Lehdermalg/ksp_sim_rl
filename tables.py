import numpy as np
from prettytable import PrettyTable

tables_folder = "episode_tables"


def reset_table():
    """Creates a PrettyTable to display results"""
    results = PrettyTable()
    table_field_names = [
        "Time [s]",
        "dt  r [u]", "cum r [u]",
        "R [m]", "φ [°]",
        "X [m]", "Y [m]",
        "V [m/s]",
        "Vr [m/s]", "Vφ [m/s]",
        "A [m/s²]",
        "Ar [m/s²]", "Aφ [m/s²]",
        "Ra [m]", "Rp [m]",
        "Throttle",
        "g(h).X [m/s²]",
        "ad(v).X [m/s²]",
        # # "g+d.X [m/s²]",
        # # "Σg [m/s]",
        # # "Σd [m/s]",
        # "TWR",
        # "Alt [m]",
        # "Fuel [u]",
        # # "Fuel tank cap [%]",
        # # "Fuel cons [u]",
        # "Mass [kg]",
        # "dV [m/s]",
        # "Thrust φ [°]"
    ]
    results.field_names = table_field_names
    results.float_format = ".3"
    results.align = "r"

    return results


def add_row(table, environment, episode_cumulative_reward):
    # apo- and periapsis
    environment.ship.calculate_ra_rp(planet=environment.planet)
    environment.ship.calculate_position_r_fi_m(planet=environment.planet)
    acc_d_mps2 = (-np.sign(environment.ship.norm_velocity_mps) *
                  environment.ship.current_experienced_air_drag_mps2)

    table.add_row([
        environment.t,
        environment.step_reward, episode_cumulative_reward,
        environment.ship.position_r_fi_m[0], environment.ship.position_r_fi_m[1],
        environment.ship.position_m[0], environment.ship.position_m[1],
        environment.ship.norm_velocity_mps,
        environment.ship.velocity_r_fi_mps[0], environment.ship.velocity_r_fi_mps[1],
        np.linalg.norm(environment.ship.acceleration_mps2),
        environment.ship.acceleration_r_fi_mps2[0], environment.ship.acceleration_r_fi_mps2[1],
        environment.ship.r_apo_peri[0], environment.ship.r_apo_peri[1],
        environment.ship.throttle,
        -environment.ship.current_experienced_gravity_mps2,
        acc_d_mps2,
        # environment.ship.current_throttled_twr_u,
        # environment.ship.current_alt_m,
        # environment.ship.total_lf_u + environment.ship.total_ox_u,
        # environment.ship.total_mass_kg,
        # environment.ship.dv_remaining_mps(planet=environment.planet),
        # np.rad2deg(environment.ship.thrust_angle)
    ])
