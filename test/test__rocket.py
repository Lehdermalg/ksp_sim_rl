import unittest
import numpy as np
from numpy.testing import assert_array_equal
from copy import deepcopy

from rocket import Rockets, engine_boar
from planet import Planets


class TestPosition(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.planet = Planets.Kerbin.value
        self.rocket = Rockets.triple_boar_408t.value  # Or any other rocket type
        self.rocket._below_ground = False

    def test_valid_position_cartesian(self):
        """Test setting a valid position values above planet surface."""
        valid_positions = [
            np.array([0, -self.planet.radius_m]),
            np.array([0, self.planet.radius_m]),
            np.array([-self.planet.radius_m, 0]),
            np.array([self.planet.radius_m, 0]),
            np.array([self.planet.radius_m + 0.1, 0]),
            np.array([self.planet.radius_m * 10.0, 0]),
        ]
        for position in valid_positions:
            self.rocket.position_m = position
            self.rocket.calculate_relative_altitude_m(planet=self.planet)
            assert_array_equal(position, self.rocket.position_m,
                               f"Position should be {position}, but is {self.rocket.position_m}")

            self.assertLessEqual(0.0, self.rocket.current_alt_m,
                                 f"Altitude should be greater than {0.0}, but is {self.rocket.position_m}")

            self.assertEqual(False, self.rocket._below_ground,
                             f"Below ground should be {False}, but is {self.rocket._below_ground}, "
                             f"{self.rocket.position_m}, {self.rocket.current_alt_m}")

    def test_edge_case_position_cartesian(self):
        """Test setting a valid throttle value."""
        edge_valid_positions = [
            np.array([0.0, -(self.planet.radius_m-0.1)]),
            np.array([0.0, (self.planet.radius_m-0.1)]),
            np.array([(self.planet.radius_m-0.1), 0.0]),
            np.array([-(self.planet.radius_m-0.1), 0.0]),
            np.array([0.0, 0.0]),
        ]
        for position in edge_valid_positions:
            self.rocket.position_m = position
            self.rocket.calculate_relative_altitude_m(planet=self.planet)
            assert_array_equal(position, self.rocket.position_m,
                               f"Position should be {position}, but is {self.rocket.position_r_fi_m}")

            self.assertGreaterEqual(0.0, self.rocket.current_alt_m,
                                    f"Altitude should be greater than {0.0}, but is {self.rocket.position_m}")

            self.assertEqual(True, self.rocket._below_ground,
                             f"Below ground should be {True}, but is {self.rocket._below_ground}")


class TestThrottle(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.planet = Planets.Kerbin.value
        self.rocket = Rockets.triple_boar_408t.value  # Or any other rocket type

    def test_valid_throttle(self):
        """Test setting a valid throttle value."""
        valid_throttles = [0.0, 0.1, 25.0, 50.0, 75.0, 99.9, 100.0]
        for throttle in valid_throttles:
            self.rocket.throttle = throttle
            self.assertEqual(throttle, self.rocket.throttle,
                             f"Throttle should be {throttle}, but is {self.rocket.throttle}")
            for eng in self.rocket.engine_list:
                self.assertEqual(throttle, eng.throttle,
                                 f"Engine throttle should be {throttle}, but is {eng.throttle}")

    def test_throttle_below_zero(self):
        """Test setting throttle below 0.0."""
        throttle = -10.0
        self.rocket.throttle = throttle
        self.assertEqual(0.0, self.rocket.throttle,
                         "Throttle should be clipped to 0.0")
        for eng in self.rocket.engine_list:
            self.assertEqual(0.0, eng.throttle,
                             f"Engine throttle should be {0.0}, but is {eng.throttle}")

    def test_throttle_above_hundred(self):
        """Test setting throttle above 100.0."""
        throttle = 110.0
        self.rocket.throttle = throttle
        self.assertEqual(100.0, self.rocket.throttle,
                         "Throttle should be clipped to 100.0")
        for eng in self.rocket.engine_list:
            self.assertEqual(100.0, eng.throttle,
                             f"Engine throttle should be {100.0}, but is {eng.throttle}")

    def test_throttle_wrong_type(self):
        """Test setting throttle above 100.0."""
        invalid_throttles = [None, "fubar", [1, 2, 3], np.array([1.00, 2.00])]
        for throttle in invalid_throttles:
            with self.assertRaises(TypeError):
                self.rocket.throttle = throttle


class TestFuel(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.planet = Planets.Kerbin.value
        self.rocket = deepcopy(Rockets.triple_boar_408t.value)  # Or any other rocket type
        self.rocket.thrust_angle = np.deg2rad(0.0)
        self.rocket.position_m = np.array([self.planet.radius_m+10.0, 0.0])
        self.rocket.velocity_mps = np.array([0.0, 0.0])
        self.rocket.acceleration_mps2 = np.array([0.0, 0.0])
        self.rocket.calculate_properties(self.planet)
        print(f"Rocket total lf u: {self.rocket.total_lf_u}")
        print(f"Rocket total f kg: {self.rocket.total_fuel_mass_kg}")
        print(f"Rocket position: {self.rocket.position_m}")

    def test_fuel_amount(self):
        """Test fuel values."""
        _stacks = 3
        _kgpu = 5
        lf_u = 2880.0 + 2880.0 + 2880.0 + 1440.0
        lf_kg_total = _stacks * lf_u * _kgpu
        ox_u = 3520.0 + 3520.0 + 3520.0 + 1760.0
        ox_kg_total = _stacks * ox_u * _kgpu
        self.assertEqual(lf_kg_total + ox_kg_total, self.rocket.initial_fuel_mass_kg,
                         f"Fuel mass should be {lf_kg_total+ox_kg_total}, but is {self.rocket.initial_fuel_mass_kg}")

    def test_fuel_consumption(self):
        """Test fuel values."""
        _throttle_value_list = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        _dt_value_list = [1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 5.0e-1, 1.0e0, 2.0e0]
        _stacks = 3
        _kgpu = 5
        boar = deepcopy(engine_boar)
        _f_consumption_single_engine_ups = boar.lf_consumption_ups + boar.ox_consumption_ups
        _f_consumption_single_engine_kgps = _f_consumption_single_engine_ups * _kgpu
        _f_consumption_stacked_kgps = _f_consumption_single_engine_kgps * _stacks

        for thr in _throttle_value_list:
            for dt in _dt_value_list:
                with (self.subTest(thr=thr, dt=dt)):
                    self.rocket.throttle = thr
                    self.rocket.position_m = np.array([self.planet.radius_m + 10.0, 0.0])

                    _thr_p = thr/100.0
                    _f_consumption_stacked_throttled_kgps = _f_consumption_stacked_kgps * _thr_p
                    _f_consumption_stacked_throttled_timed_kgps = _f_consumption_stacked_throttled_kgps * dt

                    self.rocket.calculate_throttled_fuel_consumption_u(dt=dt)
                    _expected = round(
                        (boar.lf_consumption_ups +
                         boar.ox_consumption_ups) * _stacks * dt * _thr_p,
                        6
                    )
                    _obtained = round(
                        self.rocket.throttled_lf_consumption_u +
                        self.rocket.throttled_ox_consumption_u,
                        6
                    )
                    self.assertEqual(_expected, _obtained,
                                     f"Throttled fuel consumption should be {_expected}, "
                                     f"but is {_obtained}\n"
                                     f"Params: throttle: {thr}, dt: {dt}")

                    _f_kg_t0 = self.rocket.total_fuel_mass_kg
                    self.rocket.make_force_iteration(dt=dt, planet=self.planet)
                    self.rocket.calculate_properties(self.planet)
                    _f_kg_t1 = self.rocket.total_fuel_mass_kg

                    _expected = round(_f_consumption_stacked_throttled_timed_kgps, 6)
                    _obtained = round(_f_kg_t0-_f_kg_t1, 6)

                    self.assertEqual(_expected, _obtained,
                                     f"Consumed fuel mass should be {_expected}, "
                                     f"but is {_obtained}\n"
                                     f"Params: throttle: {thr}, dt: {dt}")


if __name__ == '__main__':
    unittest.main()
