import optuna
from atu.tuner import Tuner

tuner = Tuner(
    script="atu/sac.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "Safe-StaticObstacle3d-v0": [-1000, 100],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_loguniform("learning-rate", 0.0003, 0.003),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.99),
        "batch-size": trial.suggest_categorical("batch-size", [256, 512, 1024, 2048]),
        "total-timesteps": 100000,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    wandb_kwargs={"project": "atu3"},
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)