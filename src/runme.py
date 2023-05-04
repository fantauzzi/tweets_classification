import matplotlib.pyplot as plt
import numpy as np
import optuna as opt
import transformers
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy import sin, sqrt
from optuna import Trial
from optuna.samplers import TPESampler

seed = 31415

transformers.set_seed(seed)


def g(x, y):
    z = sin(sqrt(x ** 2 + y ** 2)) / (sqrt(x ** 2 + y ** 2)) if np.linalg.norm(x) != 0 or np.linalg.norm(y) else 1.
    return z


def f(trial: Trial):
    x = trial.suggest_float("x", -10., 10.)
    y = trial.suggest_float("y", -10., 10.)
    z = g(x, y)
    return z


"""
def hp_space(trial: opt.trial.Trial) -> dict:
    res = {
        "x": trial.suggest_float("x", -10., 10.),
        "y": trial.suggest_float("y", -10., 10.),
    }
    return res
"""


def plot_it():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    Z = g(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


study_name = 'optuna_test'
trials_storage = f'sqlite:///../db/{study_name}.db'
sampler = TPESampler(seed=seed)
study = opt.create_study(direction='maximize',
                         study_name=study_name,
                         storage=trials_storage,
                         load_if_exists=True,
                         sampler=sampler)
study.optimize(f, n_trials=100)
pass
