import matplotlib.pyplot as plt
import optuna
from numpy import sin, sqrt

plt.ion()


"""
def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    z = sin(sqrt(x ** 2 + y ** 2)) / (sqrt(x ** 2 + y ** 2))
    return z
    """

def objective(trial):
    x = trial.suggest_float("x", -2, 2)
    y = trial.suggest_float("y", -2, 2)
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


optuna.logging.set_verbosity(optuna.logging.WARNING)
direction = 'minimize'
# study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler())
# study = optuna.create_study(direction=direction, sampler=optuna.samplers.QMCSampler())
# study = optuna.create_study(direction=direction, sampler=optuna.samplers.CmaEsSampler())
study = optuna.create_study(direction=direction, sampler=optuna.samplers.NSGAIISampler())
print(f"Sampler is {study.sampler.__class__.__name__}")
study.optimize(objective, n_trials=1000, show_progress_bar=True)

print(study.best_params)

trials_x = [trial.params['x'] for trial in study.trials]
trials_y = [trial.params['y'] for trial in study.trials]

dur = .01
bunch = 10
pause = dur / len(trials_x)
fig = None
for i in range(0, len(trials_x), bunch):
    plt.scatter(trials_x[i:i + bunch], trials_y[i:i + bunch])
    plt.pause(pause)

plt.ioff()
plt.show()
