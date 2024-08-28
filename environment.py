import gym
import gym.spaces as spaces
import numpy as np
from copy import deepcopy

from planet import Planets, Planet
from rocket import Rockets, Rocket
from agents import THROTTLE_ACTIONS, ANGLE_ACTIONS, FlightReplayBuffer

OBSERVATION_NAMES = [
    "radial position",
    "angular position",
    "throttle",
    "thrust angle",
    "radial velocity",
    "angular velocity",
    "radial acceleration",
    "angular acceleration",
    "apoapsis",
    "periapsis",
    "air drag acceleration",
    "fuel",
    "delta V"
]


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
    _alt_max = 90.0e+3         # Altitude limiting the trajectory for 0-1 scaling
    _fi_max = 360               # Angle limiting the trajectory for 0-1 scaling
    _v_max = 4.0e+3             # Velocity limiting the trajectory for 0-1 scaling
    _a_max = 5e+1               # Acceleration limiting the trajectory for 0-1 scaling
    _dv_max = 10.0e+3             # delta V limiting the observation space for 0-1 scaling

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
        # Create a replay buffer
        self.flight_replay_buffer = FlightReplayBuffer(max_size=int(1.0e+4))  # Adjust max_size as needed

        # This will be needed here as some of the observation space is relative to the actors (planet and rocket)
        self.reset()

        # Define the number of discrete actions
        self.n_throttle_actions = len(THROTTLE_ACTIONS)
        self.n_angle_actions = len(ANGLE_ACTIONS)
        self.action_space = spaces.Discrete(self.n_throttle_actions * self.n_angle_actions)
        self.action_space_t = spaces.Discrete(self.n_throttle_actions)
        self.action_space_a = spaces.Discrete(self.n_angle_actions)

        self._last_crash_punishment = None
        self._current_crash_punishment = None
        self.crash_punishment = None

        # observation space should be the rocket and flight parameters ... I am a dumb fool :D :D :D
        """ The NEW observation space (uncommented)
        "R [m]", "φ [°]",
        "Thr [%]", "Thr φ [°]"
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
                -0.5,  # φ [°] (angle)
                0.0,   # Throttle
                -1.0,   # Thrust angle
                -1.0,  # Vr [m/s] (radial velocity)
                -1.0,  # Vφ [m/s] (angular velocity)
                -1.0,  # Ar [m/s²] (radial acceleration)
                -1.0,  # Aφ [m/s²] (angular acceleration)
                -1.0,  # Ra [m] (apoapsis)
                -1.0,  # Rp [m] (periapsis)
                -1.0,  # ad(v).X [m/s²] (air drag acceleration 0-1 scaled between -50m/s²..0m/s²)
                0.0,  # Fuel [u]
                0.0,  # dV [m/s]
            ]),
            high=np.array([
                1.0,  # R [m] (radial distance 0-1 scaled up to 90km)
                0.5,  # φ [°] (angle 0-1 scaled between -180..180°)
                1.0,  # Throttle [ ] (% 0-1 scaled)
                1.0,  # Thrust angle [°] (angle 0-1 scaled between -180..180°)
                1.0,  # Vr [m/s] (radial velocity 0-1 scaled between -4km/s..+4km/s)
                1.0,  # Vφ [m/s] (angular velocity 0-1 scaled between -4km/s..+4km/s)
                1.0,  # Ar [m/s²] (radial acceleration 0-1 scaled between -50m/s²..+50m/s²)
                1.0,  # Aφ [m/s²] (angular acceleration 0-1 scaled between -50m/s²..+50m/s²)
                1.0,  # Ra [m] (apoapsis 0-1 scaled between -1.3kkm..+1.3kkm)
                1.0,  # Rp [m] (periapsis 0-1 scaled between -1.3kkm..+1.3kkm)
                0.0,  # ad(v).X [m/s²] (air drag acceleration)
                1.0,  # Fuel [u]
                1.0,  # dV [m/s] (delta V 0-1 scaled between 0m/s..+10km(s)
            ]),
            dtype=np.float32
        )

    def set_target_altitude_m(self, altitude: float):
        """Sets the target altitude for the ship and the 'safe' copy"""
        if altitude <= 0.0:
            print(f"The altitude must be > 0m")
            return
        self._ship.target_alt_m = altitude
        self.ship.target_alt_m = altitude
        print(f"Ship target altitude = {self.ship.target_alt_m}")

    def set_target_velocity_mps(self, velocity: float):
        """Sets the target altitude for the ship and the 'safe' copy"""
        self._ship.target_velocity_mps = velocity
        self.ship.target_velocity_mps = velocity
        print(f"Ship target velocity = {self.ship.target_velocity_mps}")

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
        self.flight_replay_buffer.reset()
        # Reset the necessary ship parameters
        # TODO: add some exploration of the initial position space
        self.ship.position_m = position_m
        self.ship.velocity_mps = velocity_mps
        self.ship.acceleration_mps2 = acceleration_mps2
        self.ship.initial_dV_mps = self.ship.dv_remaining_mps(self.planet)
        self.ship.calculate_properties(self.planet)
        print(f"Ship t0 altitude: {self.ship.current_alt_m}")

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
            # R [m] (radial distance) - calculated from kerbin surface up to 90km scaled down to 0-1 range
            # φ [°] (angle)
            (self.ship.position_r_fi_m[0] - self.planet.radius_m) / self._alt_max,
            self.ship.position_r_fi_m[1] / self._fi_max,
            # Throttle [ ] (% 0-1 scaled)
            # Thrust angle [°] (angle 0-1 scaled between -180..180°)
            self.ship.throttle / 100.0,
            self.ship.thrust_angle / self._fi_max,

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
            self.ship.dv_remaining_mps(planet=self.planet) / self._dv_max
        ], dtype=np.float32)
        # print(obs)
        return obs

    def step(self, action: np.array) -> (np.array, float, bool, dict):
        """Apply the action, update the environment, and return (obs, reward, done, info)."""
        if self.done:
            print("Evaluation terminated!")
            return None, None, self.done, {}

        throttle_delta, thrust_angle_delta = action

        # Apply action:
        self.ship.throttle += float(throttle_delta)
        # this was an early attempt to get rid of the agent smashing the rocket against the ground
        # self.ship.thrust_angle = thrust_angle + np.deg2rad(180.0)
        self.ship.thrust_angle += thrust_angle_delta
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
        from maths import gaussian
        self.step_reward = 0.0

        # weights for the rewards' importance
        # alt_rew_w = 3.0e-3
        alt_rew_w = 1.0e+2
        # vel_rew_w = 1.0e-3
        vel_rew_w = 1.0e+2
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
        velocity_reward = vel_rew_w * gaussian(
            value=self.ship.velocity_r_fi_mps[1],
            target_value=self.ship.target_velocity_mps,
            std_dev=0.05 * self.ship.target_velocity_mps)
        _total_fuel_percentage = self.ship.total_fuel_mass_kg / self.ship.initial_fuel_mass_kg
        fuel_reward = fuel_rew_w * pow(self.ship.total_fuel_mass_kg / self.ship.initial_fuel_mass_kg, 2)
        # print(f"fuel     and reward: "
        #       f"{round(_total_fuel_percentage, self.dec):{self.wid}.{self.dec}f}\t"
        #       f"{round(fuel_reward, self.dec):{self.wid}.{self.dec}f}")

        self.step_reward += altitude_reward + velocity_reward + fuel_reward + time_reward

        # crashing is really hyper ultra super bad
        # crash_punishment = round(-2.5e+5, self.dec)
        # crash_punishment = round(-2.0e+6, self.dec)  # 200 steps * 1000 reward
        # crash_punishment = round(-2.0e+6, self.dec)  # 200s * 100 steps/s * 1000 reward * 1/10
        # crash_punishment = penalty_function(self.last_epoch_mean_rewards)
        if self.ship.crashed:
            if self.crash_punishment is None:
                self.crash_punishment = -1.1 * self.cumulative_rewards[-1]
            self.step_reward += self.crash_punishment
            # adapt the punishment for a new run
            self.crash_punishment = -1.1 * self.cumulative_rewards[-1]

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
