import gym
import gym.spaces as spaces
import numpy as np
from copy import deepcopy

from planet import Planets, Planet
from rocket import Rockets, Rocket


class SimpleKSPEnv(gym.Env):
    """
        Provides a simplified Kerbal Space Program environment to teach agents how to fly rockets.
        Simplifications:
            - 2D polar coordinates
            - no lift calculation
            - no angular momentum calculation for rocket (point-like thrust)
            - no thermal calculation
            - no air forces calculations
    """

    t = 0
    dt = 0.0
    done = False
    episode_rewards = 0.0
    _ship, _planet = None, None  # To be used for resetting
    ship, planet = None, None  # To be used for calculations
    dec = 6
    wid = 9
    _alt_max = 250.0e+3         # Altitude limiting the trajectory for 0-1 scaling
    _fi_max = 360               # Angle limiting the trajectory for 0-1 scaling
    _v_max = 4.0e+3             # Velocity limiting the trajectory for 0-1 scaling
    _a_max = 3e+2               # Acceleration limiting the trajectory for 0-1 scaling

    def __init__(
            self,
            going_to_orbit=True,
            dt: float = 1.0e-2,  # time step size in seconds
            # total_time_s: int = 10,  # time cutoff in seconds - for trial runs
            total_time_s: int = 4000,  # time cutoff in seconds
            planet: Planet = Planets.Kerbin.value,
            ship: Rocket = Rockets.triple_boar_408t.value,
    ):
        super(SimpleKSPEnv, self).__init__()
        self.going_to_orbit = going_to_orbit
        self.total_time_s = total_time_s
        self.dt = dt
        # Store info for resetting
        self._planet = planet
        self._ship = ship

        # This will be needed here as some of the observation space is relative to the actors (planet and rocket)
        self.reset()

        # 1. Define action and observation spaces:
        self.action_space = spaces.Box(
            low=np.array([
                # - Throttle (continuous 0.0 to 100.0)
                0.0,
                # - Thrust Angle (continuous -pi to pi)
                -np.pi  # TODO: The thrust angle should ideally be limited to -3°..3° relative to rocket direction
            ]),
            high=np.array([
                100.0,
                np.pi
            ]),
            dtype=np.float32
        )

        # observation space should be the rocket and flight parameters ... I am a dumb fool :D :D :D
        """ The NEW observation space (uncommented)
        "R [m]", "φ [°]",
        "Vr [m/s]", "Vφ [m/s]",
        "Ar [m/s²]", "Aφ [m/s²]",
        "Ra [m]", "Rp [m]",
        "Throttle",
        "ad(v).X [m/s²]",
        "Fuel [u]",
        "dV [m/s]",
        # "g(h).X [m/s²]",
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
        """
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,  # R [m] (radial distance)
                -np.pi,  # φ [°] (angle)
                -10e+3,  # Vr [m/s] (radial velocity)
                -10e+3,  # Vφ [m/s] (angular velocity)
                -1e+3,  # Ar [m/s²] (radial acceleration)
                -1e+3,  # Aφ [m/s²] (angular acceleration)
                -1e+6,  # Ra [m] (apoapsis)
                -1e+6,  # Rp [m] (periapsis)
                -1e+4,  # ad(v).X [m/s²] (air drag acceleration)
                0.0,  # Fuel [u]
                0.0,  # dV [m/s]
            ]),
            high=np.array([
                1.0e+7,  # R [m] (radial distance)
                np.pi,  # φ [°] (angle)
                10e+3,  # Vr [m/s] (radial velocity)
                10e+3,  # Vφ [m/s] (angular velocity)
                1e+3,  # Ar [m/s²] (radial acceleration)
                1e+3,  # Aφ [m/s²] (angular acceleration)
                1e+6,  # Ra [m] (apoapsis)
                1e+6,  # Rp [m] (periapsis)
                0.0,  # ad(v).X [m/s²] (air drag acceleration)
                1e+6,  # Fuel [u]
                1e+5,  # dV [m/s]
            ]),
            dtype=np.float32
        )
        """
        # The OLD observation space ...
        self.observation_space = spaces.Box(
            low=np.array([
                # - Altitude (continuous)
                self._planet.radius_m,  # for Kerbin => 600km
                # - Angular Velocity (continuous)
                -np.inf,  # TODO: Find a realistic value
                # - Orbital Velocity (continuous)
                -5000
            ]),
            high=np.array([
                self._planet.radius_m + 5 * self._planet.altitude_cutoff_m,  # for Kerbin => 600km+350km
                np.inf,  # TODO: Find a realistic value
                5000
            ]),
            dtype=np.float32
        )
        """

    def set_target_altitude_m(self, altitude: float):
        """Sets the target altitude for the ship and the 'safe' copy"""
        if altitude <= 0.0:
            print(f"The altitude must be > 0m")
            return
        self._ship.target_alt_m = altitude
        self.ship.target_alt_m = altitude
        print(f"Ship target altitude = {self.ship.target_alt_m}")

    def set_initial_position_m(self, position: np.array):
        """Sets the initial position for the ship and the 'safe' copy"""
        self._ship.position_m = position
        self.ship.position_m = position
        print(f"Ship t0 position = {self.ship.position_m}")

    def set_initial_velocity_mps(self, velocity: np.array):
        """Sets the initial velocity for the ship and the 'safe' copy"""
        self._ship.velocity_mps = velocity
        self.ship.velocity_mps = velocity
        print(f"Ship t0 velocity = {self.ship.velocity_mps}")

    def reset(
            self,
            position_m: np.array = np.array([0.0, 0.0]),
            velocity_mps: np.array = np.array([0.0, 0.0]),
            acceleration_mps2: np.array = np.array([0.0, 0.0])
    ):
        # Retrieve planet and ship from 'safe storage'
        # self.planet = deepcopy(self._planet)
        self.planet = self._planet  # There really shouldn't be anything happening to the planet...
        self.ship = deepcopy(self._ship)
        # Reset the necessary ship parameters
        # TODO: add some exploration of the initial position space
        self.ship.position_m = position_m
        self.ship.velocity_mps = velocity_mps
        self.ship.acceleration_mps2 = acceleration_mps2
        self.ship.initial_dV_mps = self.ship.dv_remaining_mps(self.planet)
        self.ship.calculate_properties(self.planet)

        # Initialize/reset other environment state variables
        self.t = 0
        self.done = False
        self.episode_rewards = []
        self.cumulative_rewards = []
        # ...

        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        acc_d_mps2 = (-np.sign(self.ship.norm_velocity_mps) *
                      self.ship.current_experienced_air_drag_mps2)

        """Extract and return observation from the environment state."""
        obs = np.array([
            # R [m] (radial distance) - calculated from kerbin surface up to 250km scaled down to 0-1 range
            # φ [°] (angle)
            (self._alt_max - (self.ship.position_r_fi_m[0] - self.planet.radius_m)) / self._alt_max,
            self.ship.position_r_fi_m[1] / self._fi_max,
            # Vr [m/s] (radial velocity)
            # Vφ [m/s] (angular velocity)
            # This might be incorrect
            self.ship.velocity_r_fi_mps[0] / self._v_max,
            self.ship.velocity_r_fi_mps[1] / self._v_max,
            # Ar [m/s²] (radial acceleration)
            # Aφ [m/s²] (angular acceleration)
            # This might be incorrect
            self.ship.acceleration_r_fi_mps2[0] / self._a_max,
            self.ship.acceleration_r_fi_mps2[1] / self._a_max,
            # Ra [m] (apoapsis)
            # Rp [m] (periapsis)
            self.ship.r_apo_peri[0] / (2.0 * self.planet.radius_m),
            self.ship.r_apo_peri[1] / (2.0 * self.planet.radius_m),
            # ad(v).X [m/s²] (air drag acceleration)
            acc_d_mps2 / self._a_max,
            # Fuel [u]
            (self.ship.total_lf_u + self.ship.total_ox_u) / self.ship.initial_fuel_mass_kg,
            # dV [m/s]
            self.ship.dv_remaining_mps(planet=self.planet) / self.ship.initial_dV_mps
        ], dtype=np.float32)
        # print(obs)
        return obs

    def step(self, action: np.array) -> (np.array, float, bool, dict):
        """Apply the action, update the environment, and return (obs, reward, done, info)."""
        if self.done:
            print("Evaluation terminated!")
            return None, None, self.done, {}

        throttle, thrust_angle = action

        # Apply action:
        self.ship.throttle = float(throttle)
        # this was an early attempt to get rid of the agent smashing the rocket against the ground
        # self.ship.thrust_angle = thrust_angle + np.deg2rad(180.0)
        self.ship.thrust_angle = thrust_angle
        # print(f"Action: Throttle={throttle:3f}, Thrust Angle={np.rad2deg(thrust_angle):3f} degrees")

        # Simulate one time step:
        self.ship.make_force_iteration(dt=self.dt, planet=self.planet)
        self.t += round(self.dt, self.dec)
        # print(f"DEBUG: Calling step... t = {round(self.t, self.dec)}")

        # Get observation after taking the step
        observation = self._get_observation()

        # Calculate reward:
        reward = self._calculate_reward()
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) == 1:
            self.cumulative_rewards.append(reward)
        else:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)

        # Check if the episode is done:
        self.done = self._is_done()

        # Optional: You can include extra information in the 'info' dictionary
        info = {}

        return observation, reward, self.done, info

    def _calculate_reward(self):
        """Define your reward function based on the environment state."""
        # Example: Reward for getting closer to target altitude
        from reward import rayleigh_heaviside_pdf
        self.step_reward = 0.0

        # weights for the rewards' importance
        # alt_rew_w = 3.0e-3
        alt_rew_w = 1.0e+4
        # vel_rew_w = 1.0e-3
        vel_rew_w = 0.0
        # fuel_rew_w = 5.0e-5
        fuel_rew_w = 0.0
        # time_reward = 1.0e-5
        # time_reward = 1.0e-1
        time_reward = 0.0

        altitude_reward = alt_rew_w * rayleigh_heaviside_pdf(
            x=self.ship.current_alt_m,
            offset=self.planet.altitude_cutoff_m,
            sigma=5e+3)
        # print(f"altitude and reward: "
        #       f"{round(self.ship.current_alt_m, self.dec):{self.wid}.{self.dec}f}\t"
        #       f"{round(altitude_reward, self.dec):{self.wid}.{self.dec}f}")
        velocity_reward = vel_rew_w * 0.0
        _total_fuel_percentage = self.ship.total_fuel_mass_kg / self.ship.initial_fuel_mass_kg
        fuel_reward = fuel_rew_w * pow(self.ship.total_fuel_mass_kg / self.ship.initial_fuel_mass_kg, 2)
        # print(f"fuel     and reward: "
        #       f"{round(_total_fuel_percentage, self.dec):{self.wid}.{self.dec}f}\t"
        #       f"{round(fuel_reward, self.dec):{self.wid}.{self.dec}f}")

        self.step_reward += altitude_reward + velocity_reward + fuel_reward + time_reward

        # crashing is really hyper ultra super bad
        # crash_punishment = round(-2.5e+5, self.dec)
        crash_punishment = round(-8.0e+6, self.dec)
        # crash_punishment = penalty_function(self.last_epoch_mean_rewards)
        if self.ship.crashed:
            self.step_reward += crash_punishment
            # pass
        # print(f"time     and reward: "
        #       f"{round(self.t, self.dec):{self.wid}.{self.dec}f}\t"
        #       f"{round(self.step_reward, self.dec):{self.wid}.{self.dec}f}")

        return self.step_reward

    def _is_done(self):
        """Check if the episode termination conditions are met."""
        # Examples:
        if self.ship.crashed:
            print(f"CRASHED - TERMINATING")
            return True
        if self.t >= self.total_time_s:
            print(f"OUT OF TIME - TERMINATING")
            return True
        # ... other conditions (reached orbit, out of fuel, etc.)
        return False

    def render(self, mode='human'):
        """(Optional) Render the environment, e.g., using matplotlib."""
        # ... (Your rendering logic)
        pass
