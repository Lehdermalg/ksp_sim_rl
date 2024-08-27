from enum import Enum
from typing import Optional
import logging
from copy import deepcopy

import numpy as np
import math
import statistics

from ship_part import RocketMotor, RocketMotorWithFuel, FuelTank, \
    engine_boar, ft_Jumbo_64, ft_x200_32
from physics import drag_mps2
from maths import normalize
from planet import Planet
from constants import *


class Rocket(object):
    # from Planet import Planet

    # Mass related
    total_mass_kg = 0.0
    initial_mass_kg = 0.0
    initial_fuel_mass_kg = 0.0
    total_fuel_mass_kg = 0.0
    total_dry_mass_kg = 0.0
    # Fuel related
    total_lf_u = 0.0
    total_lf_consumed_u = 0.0
    total_lf_consumption_ups = 0.0
    total_ox_u = 0.0
    total_ox_consumed_u = 0.0
    total_ox_consumption_ups = 0.0
    # Throttle related
    _throttle = 0.0
    throttled_lf_consumption_u = 0.0
    throttled_ox_consumption_u = 0.0
    # Altitude and position related
    current_alt_m = 0.0
    target_alt_m = 0.0
    r_apo_peri = np.array([0.0, 0.0])
    # Velocity related
    initial_dV_mps = 0.0
    target_velocity_mps = 0.0
    # Air drag related
    total_cross_section_m2 = 0.0
    average_drag_coefficient = 0.0
    current_experienced_air_density = 0.0
    current_experienced_air_drag_mps2 = 0.0
    # Acceleration related
    current_experienced_gravity_mps2 = 0.0
    # Thrust and TWR related
    current_throttled_thrust_n = 0.0
    current_total_thrust_n = 0.0
    current_throttled_twr_u = 0.0
    current_total_twr_u = 0.0
    # Cylindrical related
    position_r_fi_m = np.array([0.0, 0.0])
    velocity_r_fi_mps = np.array([0.0, 0.0])
    acceleration_r_fi_mps2 = np.array([0.0, 0.0])
    part_list = []
    engine_list = []
    fuel_tank_list = []
    crashed = False
    _below_ground = False

    def __init__(
        self,
        payload_mass_kg: float,  # Just payload mass, no fuel, no parts... just dead-weight.
        cross_section_list: list,  # ShipPart - imagine this as a list of ShipPart stacks
        part_list: list,  # ShipPart
        # PHYSICS RELATED STUFF
        position_m: Optional[np.array] = None,
        velocity_mps: Optional[np.array] = None,
        acceleration_mps2: Optional[np.array] = None,
        planet: Optional[Planet] = None
    ):
        self.crashed = False

        self.payload_mass_kg = payload_mass_kg
        self.cross_section_list = cross_section_list
        self.part_list = part_list

        self.position_m = position_m
        self.velocity_mps = velocity_mps
        self.acceleration_mps2 = acceleration_mps2

        # Angle measured relative to 0.0, 0.0 - Kerbin Center
        self.thrust_angle = 0.0
        self.calculate_average_drag_coefficient()
        self.calculate_total_cross_section()
        self.construct_engine_list()
        self.construct_fuel_tank_list()

        self.calculate_properties(planet=planet)  # takes care of handling none-planets
        self.initial_total_mass = self.total_mass_kg
        self.initial_fuel_mass_kg = self.total_fuel_mass_kg
        if planet is not None:
            self.initial_dV_mps = self.dv_remaining_mps(planet=planet)

    def calculate_position_r_fi_m(self, planet: Planet) -> None:
        """
            Calculates the position in polar coordinates relative to a planet
        :param planet:
        :return:
        """
        # position and velocity relative to selected planet
        r_rel = self.position_m - planet.position_m

        pos_r = np.linalg.norm(r_rel)
        if pos_r < 1.0e-5:
            pos_r = 0.0
            pos_fi = 0.0
            logging.warning(f"Polar: Epsilon norm detected")
        else:
            pos_fi = 180.0*np.arcsin(r_rel[1] / pos_r)/np.pi

        self.position_r_fi_m = np.array([pos_r, pos_fi])
        if self.position_r_fi_m[0] < planet.radius_m:
            logging.warning(f"Polar: Below ground detected!")
            self._below_ground = True

    @property
    def norm_velocity_mps(self):
        """Normalizes the velocity [m/s]"""
        return np.linalg.norm(self.velocity_mps)

    def calculate_velocity_r_fi_mps(self, planet: Planet) -> None:
        """Calculates the velocity in polar coordinates relative to a planet"""
        # position and velocity relative to selected planet
        r_rel = self.position_m - planet.position_m
        # rx, ry = self.position_m[0] - planet.position_m[0], \
        #          self.position_m[1] - planet.position_m[1]
        v_rel = self.velocity_mps - planet.velocity_mps
        # vx, vy = self.velocity_mps[0] - planet.velocity_mps[0], \
        #          self.velocity_mps[1] - planet.velocity_mps[1]
        # position and velocity relative to selected planet
        r = np.linalg.norm(r_rel)
        if r < 1.0e-3:
            # r = 0.0
            logging.warning(f"Polar: Epsilon norm detected")
            self.velocity_r_fi_mps = np.array([0.0, 0.0])
            return

        self.velocity_r_fi_mps = np.array([
            (r_rel[0] * v_rel[0] + r_rel[1] * v_rel[1]) / r,
            (r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]) / r
        ])

    @property
    def norm_acceleration_mps2(self):
        """Normalizes the acceleration [m/s²]"""
        return np.linalg.norm(self.acceleration_mps2)

    def calculate_acceleration_r_fi_mps2(self, planet: Planet) -> None:
        """Calculates the acceleration in polar coordinates relative to a planet"""
        # position and acceleration relative to selected planet
        r_rel = self.position_m - planet.position_m
        a_rel = self.acceleration_mps2  # planet has no acceleration?
        r = np.linalg.norm(r_rel)
        if r < 1.0e-3:
            r = 0.0
            logging.warning(f"Polar: Epsilon norm detected")
            self.acceleration_r_fi_mps2 = np.array([0.0, 0.0])
            return

        self.acceleration_r_fi_mps2 = np.array([
            (r_rel[0] * a_rel[0] + r_rel[1] * a_rel[1]) / r,
            (r_rel[0] * a_rel[1] - r_rel[1] * a_rel[0]) / r
        ])

    def dv_remaining_mps(self, planet: Planet) -> float:
        """Calculates the remaining delta-v [m/s] relative to the planet"""
        from math import log

        avg_isp = 0.0
        isp_list = [engine.isp_asl for engine in self.engine_list]
        avg_isp = statistics.mean(isp_list)
        # print(f"DEBUG: avg_isp {avg_isp}\n"
        #       f"total_mass:{self.total_mass_kg}\n"
        #       f"total_dry_mass:{self.total_dry_mass_kg}")
        if self.total_mass_kg == 0.0:
            return 0.0
        if self.total_dry_mass_kg == 0.0:
            return 0.0
        return avg_isp * planet.gravitational_acceleration(0) * log(self.total_mass_kg / self.total_dry_mass_kg)

    @property
    def consumes_lf(self) -> bool:
        """Does the ship consume liquid fuel?"""
        engines_using_lf = [e for e in self.engine_list if e.throttle > 0.0 and
                            e.lf_consumption_ups > 0.0]
        return len(engines_using_lf) > 0

    @property
    def consumes_ox(self) -> bool:
        """Does the ship consume oxidizer?"""
        engines_using_ox = [e for e in self.engine_list if e.throttle > 0.0 and
                            e.ox_consumption_ups > 0.0]
        return len(engines_using_ox) > 0

    def calculate_total_mass_kg(self) -> None:
        """Calculates the ship total mass [kg]"""
        self.total_mass_kg = 0.0
        # parts mass
        for part in self.part_list:
            self.total_mass_kg += part.total_mass_kg
        # payload and other junk
        self.total_mass_kg += self.payload_mass_kg

    def calculate_total_lf_u(self) -> None:
        """Calculates the ship total liquid fuel [u]"""
        self.total_lf_u = 0.0
        for e in self.fuel_tank_list:
            self.total_lf_u += e.lf_u

    def calculate_total_ox_u(self) -> None:
        """Calculates the ship total oxidizer fuel [u]"""
        self.total_ox_u = 0.0
        for e in self.fuel_tank_list:
            self.total_ox_u += e.ox_u

    def calculate_total_fuel_mass_kg(self) -> None:
        """Calculates the ship total fuel mass [kg]"""
        self.total_fuel_mass_kg = (self.total_lf_u * liquid_fuel_density_kgpu +
                                   self.total_ox_u * oxidizer_density_kgpu)

    def calculate_dry_mass_kg(self) -> None:
        """Calculates the ship total mass without fuel - aka 'dry' mass [kg]"""
        self.total_dry_mass_kg = self.total_mass_kg - self.total_fuel_mass_kg

    def calculate_current_gravity(self, planet: Planet) -> None:
        """Calculates the current experienced gravity from a planet [m/s²]"""
        self.current_experienced_gravity_mps2 = planet.gravitational_acceleration(altitude_m=self.current_alt_m)

    def calculate_total_thrust_n(self) -> None:
        """Calculates the ships current total thrust [N]"""
        self.current_total_thrust_n = 0.0

        # There is thrust only if there's fuel to be burnt...
        if self.consumes_lf and self.total_lf_u <= 0.0:
            return
        if self.consumes_ox and self.total_ox_u <= 0.0:
            return

        # FOR SIMPLICITY Assumes that all rocket motors are pointing in the same direction
        for engine in self.engine_list:
            self.current_total_thrust_n += engine.thrust_n(air_density_kgpm3=self.current_experienced_air_density)

    def calculate_throttled_thrust_n__OLD(self) -> None:
        """Calculates the current total thrust including the throttle [N]"""
        self.current_throttled_thrust_n = 0.0
        if self.throttle == 0.0:
            return

        self.current_throttled_thrust_n = self.current_total_thrust_n * self.throttle/100.0

    def calculate_throttled_thrust_n(self, planet: Planet) -> None:
        """Calculates the current total thrust including the throttle [N]"""
        self.current_throttled_thrust_n = 0.0
        if self.throttle == 0.0:
            return

        air_density_kgpm3 = planet.air_density_kgpm3(altitude_m=self.current_alt_m)
        for engine in self.engine_list:
            self.current_throttled_thrust_n += (engine.thrust_n(air_density_kgpm3=air_density_kgpm3) *
                                                engine.throttle/100.0)

    def calculate_total_twr_u(self) -> None:
        """Calculates the ships current total Thrust To Weight Ratio - aka. TWR [u]"""
        self.current_total_twr_u = (self.current_total_thrust_n /
                                    self.current_experienced_gravity_mps2 /
                                    self.total_mass_kg)

    def calculate_throttled_twr_u(self) -> None:
        """Calculates the ships current total throttled TWR [u]"""
        self.current_throttled_twr_u = self.current_total_twr_u * self.throttle/100.0

    def calculate_total_cross_section(self) -> None:
        """Calculates the ships total cross-section [m²]"""
        self.total_cross_section_m2 = 0.0
        for part in self.cross_section_list:
            self.total_cross_section_m2 += math.pi * pow(part.size.value, 2.0)

    def calculate_average_drag_coefficient(self) -> None:
        """Calculates the ships average drag coefficient [unit less]"""
        drag_coefficient_list = [part.drag for part in self.cross_section_list]
        self.average_drag_coefficient = statistics.mean(drag_coefficient_list)

    def calculate_relative_altitude_m(self, planet: Planet) -> None:
        """Calculates the ships relative altitude over the planet [m]"""
        self.current_alt_m = round(np.linalg.norm(self.position_m - planet.position_m) - planet.radius_m, 3)
        if self.current_alt_m < 0.0:
            print(self.current_alt_m)
            logging.warning(f"Cartesian: Below ground detected!")
            self._below_ground = True

    def calculate_experienced_air_density(self, planet: Planet) -> None:
        """Calculates the ships experienced air density [kg/m^3]"""
        self.current_experienced_air_density = planet.air_density_kgpm3(altitude_m=self.current_alt_m)

    def calculate_experienced_air_drag(self) -> None:
        """Calculates the ships currently experienced air drag [m/s²]"""
        self.current_experienced_air_drag_mps2 = drag_mps2(
            air_density_kgpm3=self.current_experienced_air_density,
            velocity_mps=self.norm_velocity_mps,
            drag_coefficient=self.average_drag_coefficient,
            cross_section_area_m2=self.total_cross_section_m2,
            mass_kg=self.total_mass_kg
        )

    def construct_engine_list(self) -> None:
        """Constructs the engine list"""
        self.engine_list = [
            part for part in self.part_list if
            isinstance(part, (RocketMotor, RocketMotorWithFuel))
        ]

    def construct_fuel_tank_list(self) -> None:
        self.fuel_tank_list = [
            part for part in self.part_list if
            isinstance(part, (FuelTank, RocketMotorWithFuel))
        ]

    def calculate_total_fuel_consumption_ups(self) -> None:
        """Calculates the ships total fuel consumption [u/s]"""
        self.total_lf_consumption_ups = 0.0
        self.total_ox_consumption_ups = 0.0

        for e in self.engine_list:
            self.total_lf_consumption_ups += e.lf_consumption_ups
            self.total_ox_consumption_ups += e.ox_consumption_ups

    def calculate_throttled_fuel_consumption_u(self, dt: float) -> None:
        """Calculates the ships throttled fuel consumption [u]"""
        self.throttled_lf_consumption_u = 0.0
        self.throttled_ox_consumption_u = 0.0

        for e in self.engine_list:
            self.throttled_lf_consumption_u += e.throttle/100.0 * e.lf_consumption_ups
            self.throttled_ox_consumption_u += e.throttle/100.0 * e.ox_consumption_ups

        self.throttled_lf_consumption_u *= dt
        self.throttled_ox_consumption_u *= dt

    @property
    def throttle(self) -> float:
        """Gets the ships throttle value"""
        return self._throttle

    @throttle.setter
    def throttle(self, v: float) -> None:
        """Sets the ships throttle between 0-100%"""
        # Take care of wrong data types.
        if not isinstance(v, (int, float)):
            raise TypeError(f"Throttle value must be an int or ideally float: {v}")

        # Take care of out-of-bound values
        if v < 0.0:
            v = 0.0
            # logging.warning(f"Clipping throttle to {v}!")
        if v > 100.0:
            v = 100.0
            # logging.warning(f"Clipping throttle to {v}!")

        # Save data and pass it on to engines.
        self._throttle = v
        for e in self.engine_list:
            e.throttle = v

    @property
    def thrust_angle(self) -> float:
        """Gets the ships thrust angle"""
        return self._thrust_angle

    @thrust_angle.setter
    def thrust_angle(self, v: float) -> None:
        """Sets the ships thrust angle between 0.0° .. 360°"""
        # Take care of wrong data types.
        if not isinstance(v, float):
            raise TypeError(f"Thrust angle value must be a float: {v}")

        # Take care of out-of-bound values
        if v < 0.0 or v > 360.0:
            v = v % 360.0
            logging.warning(f"Removing redundant thrust angle {v}!")

        # Save data and pass it on to engines.
        self._thrust_angle = v

    @property
    def engines_operational(self) -> bool:
        """Checks if engines are operational (still have enough fuel for this time tick)"""
        return (self.consumes_lf and self.total_lf_u > 0.0 and
                self.consumes_ox and self.total_ox_u > 0.0)

    def calculate_total_acceleration_mps2(self, planet: Planet) -> None:
        """Calculates the ships total acceleration relative to the planet [m/s²]"""
        from math import sin, cos

        # PLANET GRAVITATIONAL ACCELERATION PART
        # print(f"DEBUG: position diff:{self.position_m - planet.position_m}")
        _gravity_versor = normalize(self.position_m - planet.position_m)
        # print(f"DEBUG: velocity:{self.velocity_mps}")
        _velocity_versor = normalize(self.velocity_mps)
        self.acceleration_mps2 = \
            - self.current_experienced_gravity_mps2 * _gravity_versor \
            - self.current_experienced_air_drag_mps2 * _velocity_versor

        # SHIP ENGINE ACCELERATION PART
        if not self.engines_operational:
            # Quit calculation (engine thrust)
            # if engines are off or can't consume fuel
            return

        # self.acceleration_mps2 += rocket_thrust / rocket_mass
        _fi = np.deg2rad(self.thrust_angle)
        rot = np.array([[cos(_fi), -sin(_fi)],
                        [sin(_fi),  cos(_fi)]])
        # print(f"fi = {_fi}\nrot matrix = {rot}")

        _thrust_versor = normalize(np.dot(rot, _gravity_versor))
        # IMPORTANT: Simplification - the thrust versor neglects any aerodynamics at this point

        self.acceleration_mps2 += self.current_throttled_thrust_n / self.total_mass_kg * _thrust_versor

    def calculate_ra_rp(self, planet: Planet) -> None:
        """Calculates the ship APO- and PERI-apses for the current planet"""
        r = self.position_m - planet.position_m
        r_versor = np.linalg.norm(r)
        v = self.velocity_mps
        v_versor = np.linalg.norm(v)

        # This code is NOT unreachable
        ang_mom = np.cross(r, v)
        mu = planet.gravitational_parameter
        # e = r / np.linalg.norm(r) - np.cross(v, l) / g_h   # 3D for whatever future reasons...
        e = np.array([r[0]/r_versor - ang_mom * v[1]/mu,
                      r[1]/r_versor + ang_mom * v[0]/mu])
        # print(f"ecc: {e}")

        a = mu * r_versor / (2.0 * mu - r_versor * pow(v_versor, 2))
        f1 = planet.position_m
        # f2 = -2.0 * a * e     # 3D for whatever future reasons... # second-tier comment: BRO! I'm cheering for this!
        f2 = np.array([-2.0 * a * e[0],
                       -2.0 * a * e[1]])
        r_apo = 0.5*np.linalg.norm(f2-f1) + a
        r_peri = 0.5*np.linalg.norm(f2-f1) - a

        self.r_apo_peri = np.array([r_apo, r_peri])

    def consume_fuel(self, dt: float) -> None:
        """Performs a fuel consumption during the time tick"""
        # No fuel consumption if engines are off or fuel is out
        if not self.engines_operational:
            self.throttled_lf_consumption_u = 0.0
            self.throttled_lf_consumption_u = 0.0
            return

        # Less boring options...
        # consumption in [u/s] proportional to time step
        self.calculate_throttled_fuel_consumption_u(dt=dt)

        # decreasing fuel proportionally across all tanks
        for i, ft in enumerate(self.fuel_tank_list):
            # TODO: Division by zero because no fuel left
            # take liq.fuel
            tank_contribution_lf = ft.lf_u / self.total_lf_u
            tank_lf_step_consumption = self.throttled_lf_consumption_u * tank_contribution_lf
            # if enough fuel - use it
            if ft.lf_u > tank_lf_step_consumption:
                # print(f"dtu: {tank_lf_step_consumption}")
                ft.lf_u -= tank_lf_step_consumption
                self.total_lf_consumed_u += tank_lf_step_consumption
            # otherwise empty the tank to zero
            else:
                print(f"No more liquid fuel left - PART #{i}")
                ft.lf_u = 0.0

            # take oxidizer
            tank_contribution_ox = ft.ox_u / self.total_ox_u
            tank_ox_step_consumption = self.throttled_ox_consumption_u * tank_contribution_ox
            # if enough oxidizer - use it
            if ft.ox_u > tank_ox_step_consumption:
                ft.ox_u -= tank_ox_step_consumption
                self.total_ox_consumed_u += tank_ox_step_consumption
                # otherwise empty the tank to zero
            else:
                print(f"No more oxidizer left - PART #{i}")
                ft.ox_u = 0.0

    def crash(self) -> None:
        """The sad way of ending the flight of the ship"""
        self.part_list = []
        self.current_alt_m = 0.0
        self.current_throttled_thrust_n = 0.0
        self.current_total_twr_u = 0.0
        self.current_throttled_twr_u = 0.0
        self.total_mass_kg = 0.0
        self.total_dry_mass_kg = 0.0
        self.total_lf_u = 0.0
        self.total_ox_u = 0.0
        self.throttled_lf_consumption_u = 0.0
        self.throttled_ox_consumption_u = 0.0
        self.velocity_mps = np.array([0.0, 0.0])
        self.crashed = True

    def calculate_properties(self, planet: Planet) -> None:
        """Calculate all necessary ship properties during a time tick"""
        # Update planet-relative ship parameters
        if planet is not None:
            self.calculate_relative_altitude_m(planet=planet)
            self.calculate_position_r_fi_m(planet=planet)
            self.calculate_velocity_r_fi_mps(planet=planet)
            self.calculate_acceleration_r_fi_mps2(planet=planet)
            self.calculate_current_gravity(planet=planet)
            self.calculate_experienced_air_density(planet=planet)
            self.calculate_experienced_air_drag()  # indirectly uses planet data (air density)
        # Update ship-relative ship parameters
        self.calculate_total_mass_kg()
        self.calculate_total_lf_u()
        self.calculate_total_ox_u()
        self.calculate_total_fuel_mass_kg()
        self.calculate_dry_mass_kg()
        self.calculate_total_thrust_n()
        if planet is not None:
            self.calculate_throttled_thrust_n(planet=planet)
            self.calculate_total_twr_u()
            self.calculate_throttled_twr_u()

    def make_force_iteration(self, dt: float, planet: Planet) -> None:
        """
            Runs all necessary .calculate_xxx methods in necessary order
            to obtain all necessary .current_xxx values
        """
        # Update planet-relative ship parameters
        self.calculate_properties(planet=planet)

        # detect and handle the KABLAMSKI
        # if self.current_alt_m < 0.0e+3 or self.current_alt_m > 150.0e+3:
        # HIGH ALTITUDE TRAINING
        if (self.current_alt_m < self.target_alt_m - 10.0e+3 or
                self.current_alt_m > self.target_alt_m + 10.0e+3):
            self.crash()  # The ship be close to ideal orbit to learn the angular component

        # UPDATING
        #  fuel amount to produce thrust
        self.consume_fuel(dt=dt)
        #  sum up thrust and gravitational acceleration [m/s²]
        self.calculate_total_acceleration_mps2(planet=planet)
        #  calculate velocity from acceleration [m/s]
        self.velocity_mps += self.acceleration_mps2 * dt
        #  calculate position from velocity [m]
        self.position_m += self.velocity_mps * dt
        # print(f"a\tdv\tdx:\t{s_a:.2f}\t{s_dv:.2f}\t{s_dx:.2f}")


class Rockets(Enum):
    triple_boar_408t = Rocket(
        payload_mass_kg=10.0e+3,
        part_list=[
            deepcopy(engine_boar), deepcopy(ft_Jumbo_64), deepcopy(ft_Jumbo_64), deepcopy(ft_x200_32),
            deepcopy(engine_boar), deepcopy(ft_Jumbo_64), deepcopy(ft_Jumbo_64), deepcopy(ft_x200_32),
            deepcopy(engine_boar), deepcopy(ft_Jumbo_64), deepcopy(ft_Jumbo_64), deepcopy(ft_x200_32),
        ],
        # Three stacks - biggest part is the boar ... but seriously ...
        # TODO: once it starts working well enough - make the ShipParts Iterables and let the ship
        #  calculate the cross section properly ('ier')
        cross_section_list=3*[engine_boar],
    )
