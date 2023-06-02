from scipy.stats import norm


def calculate_K_mean_scaled_pos(
        rng, kriging_mean, kriging_std, current_max
):
    """Calculate scaled position of the Kriging mean on a given position related to the 
    current maximum

    Args:
        rng (list): Range of the considered input(s) in X
        kriging_mean (list): Kriging mean for all inputs in X
        kriging_std (list): Kriging standard deviation for all inputs in X
        current_max (float): Current maximum of the objective function

    Returns:
        u_x (float): scaled position of Kriging mean on all considered x
    """

    u_x = (kriging_mean[rng] - current_max) / kriging_std[rng]

    return u_x


def expected_improvement(
        rng, kriging_mean, kriging_std, current_max
):
    """Calculate potential to improve the current maximum on all considered inputs

    Args:
        rng (list): Range of the considered input(s) in X
        kriging_mean (list): Kriging mean for all inputs in X
        kriging_std (list): Kriging standard deviation for all inputs in X
        current_max (float): Current maximum of the objective function

    Returns:
        u_x (float): scaled position of Kriging mean on all considered x
    """

    u_x = calculate_K_mean_scaled_pos(rng, kriging_mean, kriging_std, current_max)
    ei_x = kriging_std[rng] * (u_x * norm.cdf(u_x) + norm.pdf(u_x))

    return ei_x
