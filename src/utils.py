import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,6)

################################ TOY OBJECTIVE FUNCTION #########################################


def obj_function(x) -> np.array:
    """toy numerical simulator - 1D"""

    return np.squeeze((x * (10 - x)) * np.sin(2 * x))


################################  EXPECTED IMPROVEMENT  #########################################


def u(kriging_mean, kriging_std, current_max):
    """Calculate scaled position of the Kriging mean on a given position related to the current
    maximum

    Args:
        kriging_mean (list): Kriging mean for all inputs in X
        kriging_std (list): Kriging standard deviation for all inputs in X
        current_max (float): Current maximum of the objective function

    Returns:
        float: scaled position of Kriging mean on all considered x
    """

    u_x = (kriging_mean - current_max) / kriging_std

    return u_x


def expected_improvement(kriging_mean, kriging_std, current_max):
    """Calculate potential to improve the current maximum on all considered inputs

    Args:
        kriging_mean (list): Kriging mean for all inputs in X
        kriging_std (list): Kriging standard deviation for all inputs in X
        current_max (float): Current maximum of the objective function

    Returns:
        int: scaled position of Kriging mean on all considered x
    """

    u_x = u(kriging_mean, kriging_std, current_max)
    ei_x = kriging_std * (u_x * norm.cdf(u_x) + norm.pdf(u_x))

    return ei_x


############################ METRIC FUNCTIONS #########################################


def mae(y, kriging_mean, x_lim, x_nbr):
    """mean absolute error metric"""

    return abs(y - kriging_mean).sum() * (x_lim[1] - x_lim[0]) / x_nbr


def rmse(y, kriging_mean, x_lim, x_nbr):
    """root mean squared error metric"""

    return (((y - kriging_mean) ** 2).sum() * (x_lim[1] - x_lim[0]) / x_nbr) ** 0.5


############################# NUMERICAL SIMULATOR ###################################


def calculate_max_EGT(oat, max_speed, altitude, humidity, air_density):
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

    T = (1.5 + 0.5 * np.sin(oat / 300 * np.pi)) * oat
    S = (1.5 + 0.5 * np.sin(oat * max_speed / 6e4)) * min(max_speed, 280)
    A = 1 - altitude * np.sin(oat / 300 * np.pi) / 5e4
    H = (1.1 - humidity * oat / 2e5) * (
        np.sin((altitude / 1e4 + max_speed / 300) * np.pi) / 5 + 1
    )
    D = 1 - air_density * np.sin(air_density) / 20

    # egt = 4 * T + S * A * H * D
    egt = ((4 * T + S * A * H * D - 2036) / (2450 - 2036) * 300) + 1000

    return egt


def _parse_time_block(seconds: int, unit: int, unit_name: str, time_str: str):

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

    egt_calculated = calculate_max_EGT(oat, max_speed, humidity, altitude, air_density)
    print(egt_calculated)

    assert egt == round(egt_calculated), "Simulator is NOT OK."
    print("Simulator is OK.")


def generate_gps(n_samples, x, x_grid, std=1, length_scale=1, y=None, show_predictor=True):

    X = x.reshape(-1, 1)

    kernel = std**2 * RBF(length_scale=length_scale, length_scale_bounds=(1e-1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=3)

    if y is not None:
        gpr.fit(X, y)
        plt.scatter(X, y, c="black", label="observations")

    y_mean, y_std = gpr.predict(x_grid.reshape(-1, 1), return_std=True)
    y_samples = gpr.sample_y(x_grid.reshape(-1, 1), n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        plt.plot(
            x_grid,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
        
    if show_predictor:
        plt.plot(x_grid, y_mean, color="black", label="Mean")
        plt.fill_between(
            x_grid,
            y_mean - 1.96*y_std,
            y_mean + 1.96*y_std,
            alpha=0.1,
            color="black",
            label=r"$\pm$ 1.96 std. dev.",
        )
            
    
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()


def generate_prior_gps(x, n_samples, std, length_scale):
    
    x_grid=x

    generate_gps(x=x, n_samples=n_samples, std=std, length_scale=length_scale, x_grid=x_grid)


def generate_posterior_gps(x,x_grid, y, n_samples, show_predictor=False):

    generate_gps(x=x, y=y, n_samples=n_samples, x_grid=x_grid, show_predictor=show_predictor)
