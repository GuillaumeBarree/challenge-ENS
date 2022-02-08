"""This module aims to define utils function for the project."""
import json
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from models.LinearNet_1 import LinearNet_1


def load_model(cfg, input_size):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["TRAIN"]["MODEL"] == "LinearNet_1":
        return LinearNet_1(num_features=input_size)
    else:
        return LinearNet_1(num_features=input_size)


def launch_grid_search(cfg, preprocessed_data):
    """Launch a grid search on different models

    Args:
        cfg (dict): Configuration file
        preprocessed_data (dict): data
    """
    # Train
    x_train = preprocessed_data["x_train"]
    y_train = preprocessed_data["y_train"]
    # Valid
    x_valid = preprocessed_data["x_valid"]
    y_valid = preprocessed_data["y_valid"]

    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        rfr = RandomForestRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(18, 28, 1),
            "max_features": np.arange(50, x_train.shape[1], 5),
            "n_estimators": np.arange(70, 120, 5),
        }

        rfr_cv = GridSearchCV(rfr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        rfr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in rfr_cv.best_params_.items():
            params[key] = int(value)

        with open("best_params_random_forest.json", "w") as outfile:
            json.dump(params, outfile, indent=2)

    elif cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":
        rfr = ExtraTreesRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(18, 28, 1),
            "max_features": np.arange(50, x_train.shape[1], 5),
            "n_estimators": np.arange(70, 120, 5),
        }

        rfr_cv = GridSearchCV(rfr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        rfr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in rfr_cv.best_params_.items():
            params[key] = int(value)

        with open("best_params_extra_trees.json", "w") as outfile:
            json.dump(params, outfile, indent=2)
