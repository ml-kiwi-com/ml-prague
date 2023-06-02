import pandas as pd
import numpy as np


def calculate_max_EGT(
    oat, max_speed, altitude, humidity, air_density
):
    """Calculate the maximal exhaust gas temperature during take off

    Args:
        oat (float): Outside air temperature at take off [K]
        max_speed (float): Maximal flight speed at take off [km/h]
        altitude (float): Airport altitude at take off [m]
        humidity (float): Air humidity at take off [%]
        air_density (float): Air density at take off [kg/m3]

    Returns:
        egt: Maximal exhaust gas temperature [K]
    """

    T = (1.5 + .5 * np.sin(oat / 300 * np.pi)) * oat
    S = (1.5 + .5 * np.sin(oat * max_speed / 6e4)) * min(max_speed, 280)
    A = (1 - altitude * np.sin(oat / 300 * np.pi) / 5e4)
    H = (1.1 - humidity * oat / 2e5) * (np.sin((altitude / 1e4 + max_speed / 300) * np.pi)/5 + 1)
    D = (1 - air_density * np.sin(air_density) / 20)

    #egt = 4 * T + S * A * H * D
    egt = ((4 * T + S * A * H * D - 2036) / (2450 - 2036) * 300) + 1000

    return egt


def _parse_time_block(
    seconds: int, unit: int, unit_name: str, time_str: str
):

    units = seconds // unit

    if units > 0:
        time_str = time_str + f"{units} {unit_name} "

    remaining_seconds = seconds - units * unit

    return time_str, remaining_seconds


def _time_format(seconds: int) -> str:

    minute = 60
    hour = 60 * minute
    day = 24 * hour
    week = 7 * day
    year = 365 * day

    time_str = ""
    time_str, seconds = _parse_time_block(seconds, year, 'years', time_str)
    time_str, seconds = _parse_time_block(seconds, week, 'weeks', time_str)
    time_str, seconds = _parse_time_block(seconds, day, 'days', time_str)
    time_str, seconds = _parse_time_block(seconds, hour, 'hours', time_str)
    time_str, seconds = _parse_time_block(seconds, minute, 'minutes', time_str)

    if seconds > 0:
        time_str = time_str + f'{seconds} seconds'

    return time_str


def simulate_max_egt(df: pd.DataFrame, fake_simulation_time: int = 3) -> pd.Series:

    consumption = [
        calculate_max_EGT(
            df.loc[i, 'oat'],
            df.loc[i, 'max_speed'],
            df.loc[i, 'altitude'],
            df.loc[i, 'humidity'],
            df.loc[i, 'air_density'],
        )
        for i in df.index
    ]
    consumption = pd.Series(consumption, index=df.index)

    num_sim = df.shape[0]
    print(f'Generated {num_sim} simulations.')

    time = fake_simulation_time * num_sim
    time_str = _time_format(time)

    print(f"Let's pretend this took {time_str}.")

    return consumption


def generate_input_data(
    num_obs: int,
    oat_min: float = 223,
    oat_max: float = 333,
    max_speed_min: float = 230,
    max_speed_max: float = 290,
    humidity_min: float = 0,
    humidity_max: float = 1,
    altitude_min: float = 0,
    altitude_max: float = 4500,
    air_density_min: float = 0.7,
    air_density_max: float = 1.6,
) -> pd.DataFrame:

    limits = {
        "oat": (oat_min, oat_max),  # world record
        "max_speed": (
            max_speed_min,
            max_speed_max,
        ),  # max for Lockheed SR-71 Blackbird
        "humidity": (humidity_min, humidity_max),
        "altitude": (altitude_min, altitude_max),
        "air_density": (air_density_min, air_density_max),
    }

    df = pd.DataFrame(index=list(range(num_obs)), columns=limits.keys())

    for var_name, var_limits in limits.items():
        var_min, var_max = var_limits
        df[var_name] = np.random.uniform(var_min, var_max, num_obs)

    return df


def test_calculate_max_EGT():

    oat = 250
    max_speed = 250
    humidity = 50
    altitude = 1000
    air_density = 0

    egt = 735

    egt_calculated = calculate_max_EGT(
        oat, max_speed, humidity, altitude, air_density
    )
    print(egt_calculated)

    assert egt == round(egt_calculated), "Simulator is NOT OK."
    print("Simulator is OK.")
