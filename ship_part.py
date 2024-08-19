from enum import Enum
from typing import Optional

from constants import liquid_fuel_density_kgpu, oxidizer_density_kgpu, liquid_fuel_cpu, oxidizer_cpu
from planet import Planets


class RadialSize(Enum):
    """Provides an enum of ship part radial sizes [m]"""
    S0_XX = 0.000
    S1_Ti = 0.625
    S2_Sm = 1.250
    S3_La = 2.500
    S4_XL = 3.750
    S5_Hu = 5.000


class ShipPart(object):
    """
        Defines ship parts. Allows storing their most important parameters like
        - name,
        - mass,
        - size,
        - drag properties
        - cost
        Later on allows to calculate total cost, total mass, etc.
    """
    def __init__(
        self,
        name: str,
        mass_kg: float,
        size: RadialSize,
        generates_drag: bool = False,
        drag: Optional[float] = None,
        cost: Optional[float] = None,
    ):
        self.name = name
        self.mass_kg = mass_kg
        self.cost = cost
        self.size = size
        self.drag = drag  # drag coefficient of this part - used only if .generates_drag
        self.generates_drag = generates_drag
        if self.generates_drag and self.drag is None:
            raise ValueError(f"Part {name} set to '.generates_drag'={self.generates_drag}, but has '.drag'={self.drag}")

    @property
    def total_mass_kg(self):
        # Intended to allow calculating other weight contributions (fuel, etc.)
        return self.mass_kg


class FuelTank(ShipPart):
    """Fuel tanks which extend the ship parts with liquid fuel / oxidizer amounts"""
    def __init__(
        self,
        name: str,
        mass_kg: float,
        lf_u: float,
        ox_u: float,
        lf_u_cap: float,
        ox_u_cap: float,
        size: RadialSize,
        cost: Optional[float] = None,
        generates_drag: bool = False,
        drag: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            mass_kg=mass_kg,
            cost=cost,
            size=size,
            drag=drag,
            generates_drag=generates_drag
        )
        if lf_u > lf_u_cap:
            raise ValueError(f"Liquid fuel amount was larger than capacity: {lf_u} > {lf_u_cap}.")
        self.lf_u_cap = lf_u_cap
        self.lf_u = lf_u

        if ox_u > ox_u_cap:
            raise ValueError(f"Oxidizer amount was larger than capacity: {ox_u} > {ox_u_cap}.")
        self.ox_u_cap = ox_u_cap
        self.ox_u = ox_u

    @property
    def total_fuel_mass_kg(self):
        return self.lf_u * liquid_fuel_density_kgpu + \
               self.ox_u * oxidizer_density_kgpu

    @property
    def total_mass_kg(self):
        return \
            self.mass_kg + \
            self.total_fuel_mass_kg

    @property
    def total_cost(self):
        return \
            self.cost + \
            self.lf_u * liquid_fuel_cpu + \
            self.ox_u * oxidizer_cpu


class RocketMotor(ShipPart):
    """Rocket motors which extend ship parts with ISP, thrust, throttle, fuel consumption"""
    def __init__(
        self,
        name: str,
        mass_kg: float,
        size: RadialSize,
        thrust_asl_n: float,
        thrust_vac_n: float,
        isp_asl: float,
        isp_vac: float,
        lf_consumption_ups: float,
        ox_consumption_ups: float,
        throttle: float = 0.0,
        generates_drag: bool = False,
        drag: Optional[float] = None,
        cost: Optional[float] = None,
    ):
        ShipPart.__init__(
            self,
            name=name,
            mass_kg=mass_kg,
            cost=cost,
            size=size,
            drag=drag,
            generates_drag=generates_drag
        )
        self.throttle = throttle
        self.thrust_vac_n = thrust_vac_n
        self.thrust_asl_n = thrust_asl_n
        self.isp_asl = isp_asl
        self.isp_vac = isp_vac
        self.lf_consumption_ups = lf_consumption_ups
        self.ox_consumption_ups = ox_consumption_ups

    def thrust_n(
        self,
        air_density_kgpm3: float
    ):
        # Provides FINAL thrust in [N], takes into account the pressure information
        # from Planet import kerbin

        return self.thrust_vac_n - (self.thrust_vac_n - self.thrust_asl_n) * \
               air_density_kgpm3 / Planets.Kerbin.value.pressure_p1


class RocketMotorWithFuel(RocketMotor, FuelTank):
    """Special class combining rocket motor with fuel tank"""
    def __init__(
        self,
        name: str,
        mass_kg: float,
        size: RadialSize,
        thrust_asl_n: float,
        thrust_vac_n: float,
        isp_asl: float,
        isp_vac: float,
        lf_consumption_ups: float,
        ox_consumption_ups: float,
        lf_u: float,
        lf_u_cap: float,
        ox_u: float,
        ox_u_cap: float,
        generates_drag: bool = False,
        drag: Optional[float] = None,
        cost: Optional[float] = None,
    ):
        RocketMotor.__init__(
            self,
            name=name,
            mass_kg=mass_kg,
            size=size,
            cost=cost,
            drag=drag,
            generates_drag=generates_drag,
            thrust_asl_n=thrust_asl_n,
            thrust_vac_n=thrust_vac_n,
            isp_asl=isp_asl,
            isp_vac=isp_vac,
            lf_consumption_ups=lf_consumption_ups,
            ox_consumption_ups=ox_consumption_ups,
        )
        FuelTank.__init__(
            self,
            name=name,
            mass_kg=mass_kg,
            size=size,
            cost=cost,
            drag=drag,
            generates_drag=generates_drag,
            lf_u=lf_u,
            lf_u_cap=lf_u_cap,
            ox_u=ox_u,
            ox_u_cap=ox_u_cap,
        )


engine_nerv = RocketMotor(
    name="LV-N Nerv Atomic Rocket Motor",
    mass_kg=3.0e+3,
    cost=10.0e+3,
    drag=0.2,
    thrust_vac_n=60.0e+3,
    thrust_asl_n=13.88e+3,
    isp_asl=185.0,
    isp_vac=800.0,
    lf_consumption_ups=1.53,
    ox_consumption_ups=0.0,
    size=RadialSize.S2_Sm,
)

engine_boar = RocketMotorWithFuel(
    name="Twin Boar Engine",
    mass_kg=10500,
    cost=14062.40,
    drag=0.2,
    thrust_vac_n=2000.000e+3,
    thrust_asl_n=1866.666e+3,
    isp_asl=280.0,
    isp_vac=300.0,
    lf_consumption_ups=135.964 * 9.0 / 20.0,
    ox_consumption_ups=135.964 * 11.0 / 20.0,
    lf_u=2880.0,
    ox_u=3520.0,
    lf_u_cap=2880.0,
    ox_u_cap=3520.0,
    size=RadialSize.S3_La,
)

ft_x200_8 = FuelTank(
    name="Rockomax X200-8 Fuel Tank",
    size=RadialSize.S3_La,
    drag=0.2,
    cost=432.8,
    mass_kg=500,
    lf_u=360,
    lf_u_cap=360,
    ox_u=440,
    ox_u_cap=440,
)

ft_x200_16 = FuelTank(
    name="Rockomax X200-16 Fuel Tank",
    size=RadialSize.S3_La,
    drag=0.2,
    cost=815.6,
    mass_kg=1000,
    lf_u=720,
    lf_u_cap=720,
    ox_u=880,
    ox_u_cap=880,
)

ft_x200_32 = FuelTank(
    name="Rockomax X200-32 Fuel Tank",
    size=RadialSize.S3_La,
    drag=0.2,
    cost=1531.2,
    mass_kg=2000,
    lf_u=1440,
    lf_u_cap=1440,
    ox_u=1760,
    ox_u_cap=1760,
)

ft_Jumbo_64 = FuelTank(
    name="Rockomax Jumbo-64 Fuel Tank",
    size=RadialSize.S3_La,
    drag=0.2,
    cost=2812.4,
    mass_kg=4000,
    lf_u=2880,
    lf_u_cap=2880,
    ox_u=3520,
    ox_u_cap=3520,
)
