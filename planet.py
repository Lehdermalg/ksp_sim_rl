import numpy as np
import math
from enum import Enum

from constants import gravitational_constant


class Planet(object):
    def __init__(
        self,
        name: str,
        mass_kg: float,
        radius_m: float,
        sidereal_orbital_period_s: float,
        sidereal_rotation_period_s: float,
        apoapsis_m: float,
        periapsis_m: float,
        position_m: np.array,
        velocity_mps: np.array,
        pressure_p1: float = 0.0,
        pressure_p2: float = 0.0,
        pressure_p3: float = 0.0,
        altitude_cutoff_m: float = 0.0
    ):
        self.name = name
        self.mass_kg = mass_kg
        self.radius_m = radius_m
        self.sidereal_orbital_period_s = sidereal_orbital_period_s
        self.sidereal_rotation_period_s = sidereal_rotation_period_s
        self.apoapsis_m = apoapsis_m
        self.periapsis_m = periapsis_m
        self.position_m = position_m
        self.velocity_mps = velocity_mps
        self.pressure_p1 = pressure_p1
        self.pressure_p2 = pressure_p2
        self.pressure_p3 = pressure_p3
        self.altitude_cutoff_m = altitude_cutoff_m

    def orbital_velocity_mps(self, altitude_m: float) -> float:
        """Calculates the orbital velocity at the given altitude [m/s]"""
        return pow(gravitational_constant * self.mass_kg / (self.radius_m + altitude_m), 0.5)

    @property
    def gravitational_parameter(self) -> float:
        """Calculates the gravitational parameter [m³/s²]"""
        return self.mass_kg * gravitational_constant

    @property
    def g(self) -> float:
        """Calculates the gravitational constant [m/s²]"""
        return self.gravitational_parameter * pow(self.radius_m, -2.0)

    def gravitational_acceleration(self, altitude_m: float) -> float:
        """Calculates the gravitational acceleration [m/s²]"""
        return self.gravitational_parameter * pow(self.radius_m + altitude_m, -2.0)

    def air_density_kgpm3(self, altitude_m: float) -> float:
        """Calculates the air density at a given altitude [kg/m³]"""
        if altitude_m >= self.altitude_cutoff_m:
            return 0.0
        # KerbinPressureParam1 * exp(-altitude_m*( KerbinPressureParam2 * KerbinGravitationalAcceleration /
        #  ( KerbinMass * GravitationalConstant / (KerbinRadius+altitude_m)^2) - KerbinPressureParam3))
        return self.pressure_p1 * math.exp(
            -altitude_m * (self.pressure_p3 + self.pressure_p2 *
                           self.g / self.gravitational_acceleration(altitude_m=altitude_m)))


class Planets(Enum):
    Kerbin = Planet(
        name="Kerbin",
        mass_kg=5.2915158e+22,
        radius_m=600.0e+3,
        sidereal_orbital_period_s=9203545,
        sidereal_rotation_period_s=21549.43,
        apoapsis_m=13599840256,
        periapsis_m=13599840256,
        position_m=np.array([0.0, 0.0]),
        velocity_mps=np.array([0.0, 0.0]),
        # velocity=np.array([9285.0, 0.0])
        pressure_p1=1.225, pressure_p2=2.0e-4, pressure_p3=-1.0/15.0e+3,
        altitude_cutoff_m=70.0e+3
    )
