def drag_mps2(
    air_density_kgpm3: float,
    velocity_mps: float,
    drag_coefficient: float,
    cross_section_area_m2: float,
    mass_kg: float
) -> float:
    """
        calculates drag in [m/s²] using:

                   rho * v² * C * A
            Fd = --------------------
                        2 * m

    :param air_density_kgpm3:
        float: air density in [kg/m³]
    :param velocity_mps:
        float: velocity in [m/s]
    :param drag_coefficient:
        float: drag coefficient depending on the shape of ship / part
    :param cross_section_area_m2:
        float: cross-section area in [m²]
    :param mass_kg:
        float: mass in [kg]

    :return:
        float: drag in [m/s²] - how much velocity [m/s] will be lost per [s] of flight
    """
    return 0.5 * air_density_kgpm3 * pow(velocity_mps, 2.0) * drag_coefficient * cross_section_area_m2 / mass_kg


def thrust_time(
        rocket: 'Rocket',
        delta_v: float
) -> float:
    """
        calculates thrust time in [s] using:

                   m0 * ISP * g0     (        [        dV      ] )
            Tt = ---------------- * (  1 - exp[ - ------------ ]  )
                        Ft           (        [     ISP * g0   ] )

    :param rocket:
        Rocket: contains necessary data like m0, ISP, Ft
    :param delta_v:
        float: velocity in [m/s]

    :return:
    """
    from planet import Planets
    import numpy as np

    m0 = rocket.total_mass_kg
    isp = rocket.engine_list[0].isp_vac                # simple calculation for vacuum
    g0 = Planets.Kerbin.value.gravitational_parameter  # engine params are given relative to Kerbin anyways
    Ft = sum([e.thrust_vac_n for e in rocket.engine_list])

    return m0 * isp * g0 / Ft * (1 - np.exp(-delta_v / (isp * g0)))
