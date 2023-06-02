import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (15,6)


def show_basic_kriging_plot_1D(
    X: np.array,
    y: np.array,
    X_train: np.array,
    y_train: np.array,
    kriging_mean: np.array,
    kriging_std: np.array,
    title: str,
):

    plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, kriging_mean, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        kriging_mean - 1.96 * kriging_std,
        kriging_mean + 1.96 * kriging_std,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title(title)
    plt.show()


def show_basic_kriging_improvement_1D(
    X: np.array,
    y: np.array,
    X_train: np.array,
    y_train: np.array,
    kriging_mean: np.array,
    kriging_std: np.array,
    new_x: np.array,
    new_y: float,
    title: str,
):

    plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.scatter(new_x, new_y, label="Added Observation", marker="*", s=200, c="purple")
    plt.plot(X, kriging_mean, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        kriging_mean - 1.96 * kriging_std,
        kriging_mean + 1.96 * kriging_std,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title(title)
    
    if "current max" in title:
        plt.scatter(X_train[y_train.argmax()],y_train.max(), marker="*", s=300, c="red")
    
    plt.show()

def show_multiD_slices_plots(X, x_predict, x_axis, gaussian_process, y_predict):
    
    for i in range(X.shape[1]):
        x_plot = np.repeat(x_predict, len(x_axis), axis=0)
        x_plot[:, i] = x_axis
        kriging_mean, kriging_std = gaussian_process.predict(x_plot, return_std=True)
        plt.scatter(x_predict[0, i], y_predict, label="Observation")
        plt.plot(x_axis, kriging_mean, label="Mean prediction")
        plt.fill_between(
            x_axis.ravel(),
            kriging_mean - 1.96 * kriging_std,
            kriging_mean + 1.96 * kriging_std,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        
        var_name = X.columns[i]
        
        plt.title(f"Kriging on Num Simulator: behavior for varying {var_name}")
        plt.show()
    