"""This module aims to define utils function for the project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.svm import NuSVR
from bayes_opt import BayesianOptimization

from models.LinearNet_1 import LinearNet_1
from models.MachineLearningModels import models


def load_model(cfg, input_size=100):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["MODELS"]["NN"]:
        if cfg["TRAIN"]["MODEL"] == "LinearNet_1":
            return LinearNet_1(num_features=input_size)
        else:
            return LinearNet_1(num_features=input_size)
    else:
        return models(cfg=cfg)


def launch_grid_search(cfg, preprocessed_data):  # pylint: disable=too-many-locals
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
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        rfr_cv = GridSearchCV(rfr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        rfr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in rfr_cv.best_params_.items():
            params[key] = int(value)

        return rfr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":
        etr = ExtraTreesRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        etr_cv = GridSearchCV(etr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        etr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in etr_cv.best_params_.items():
            params[key] = int(value)

        return etr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "GradientBoosting":
        gbr = GradientBoostingRegressor()

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 1, 0.5],
            "min_samples_leaf": [4, 5, 6],
            "subsample": [0.6, 0.7, 0.8],
            "min_samples_split": np.arange(4, 8, 2),
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        gbr_cv = GridSearchCV(gbr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        gbr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in gbr_cv.best_params_.items():
            params[key] = int(value)

        return gbr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "NuSVR":
        nusvr = NuSVR()

        param_grid = {
            "C": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
            "gamma": [0.008, 0.009, 0.01, 0.02, 0.03, "auto"],
            "kernel": ["poly", "rbf"],
            "nu": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }

        nusvr_cv = GridSearchCV(
            nusvr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2
        )
        nusvr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in nusvr_cv.best_params_.items():
            params[key] = int(value)

        return nusvr_cv.best_estimator_, params

    model = RandomForestRegressor(
        bootstrap=False,
        max_depth=22,
        max_features=50,
        min_samples_split=4,
        n_estimators=80,
        n_jobs=-1,
    )
    params = {
        "max_depth": 22,
        "max_features": 50,
        "min_samples_split": 4,
        "n_estimators": 80,
        "bootstrap": False,
        "n_jobs": 1,
    }
    return model, params


def bayesian_optimization(function, parameters, n_iterations=5):
    """Bayesian Optimisation function.
    Given the dataset and a function it optimises its hyper parameters.

    Args:
        preprocessed_data (dict): data
        function (function): function to be optimized
        parameters (dict): hyper_parameters of the model

    Returns:
        [Dict]: {'target': -4.441293113411222, 'params': {'y': y_value, 'x': x_value}}
    """
    gp_params = {"alpha": 1e-4}

    ba_op = BayesianOptimization(function, parameters)
    ba_op.maximize(n_iter=n_iterations, **gp_params)

    return ba_op.max


def launch_bayesian_opt(cfg, preprocessed_data):  # pylint: disable=too-many-locals
    """Launch a bayesian optimisation on different models

    Args:
        cfg (dict): Configuration file
        preprocessed_data (dict): data
    """
    # Train
    x_train = preprocessed_data["x_train"]
    y_train = preprocessed_data["y_train"]

    cv_splits = 4
    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":

        def rf_function(min_samples_split, max_depth, max_features, n_estimators):
            return cross_val_score(
                RandomForestRegressor(
                    n_estimators=int(max(n_estimators, 0)),
                    max_depth=int(max(max_depth, 1)),
                    min_samples_split=int(max(min_samples_split, 2)),
                    max_features=int(max(max_features, 1)),
                    n_jobs=-1,
                    random_state=42,
                ),
                X=x_train,
                y=y_train,
                cv=cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            ).mean()

        parameters = {
            "min_samples_split": (2, 10),
            "max_depth": (1, 150),
            "max_features": (10, x_train.shape[1]),
            "n_estimators": (10, 300),
        }

        best_solution = bayesian_optimization(
            rf_function,
            parameters,
            n_iterations=cfg["MODELS"]["ML"]["BAYESIAN_ITERATIONS"],
        )

        params = best_solution["params"]

        model = RandomForestRegressor(
            n_estimators=int(max(params["n_estimators"], 0)),
            max_depth=int(max(params["max_depth"], 1)),
            min_samples_split=int(max(params["min_samples_split"], 2)),
            max_features=int(max(params["max_features"], 1)),
            n_jobs=-1,
            random_state=42,
        )

        return model, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":

        def etr_function(min_samples_split, max_depth, max_features, n_estimators):
            return cross_val_score(
                ExtraTreesRegressor(
                    n_estimators=int(max(n_estimators, 0)),
                    max_depth=int(max(max_depth, 1)),
                    min_samples_split=int(max(min_samples_split, 2)),
                    max_features=int(max(max_features, 1)),
                    n_jobs=-1,
                    random_state=42,
                ),
                X=x_train,
                y=y_train,
                cv=cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            ).mean()

        parameters = {
            "min_samples_split": (2, 10),
            "max_depth": (1, 300),
            "max_features": (10, x_train.shape[1]),
            "n_estimators": (10, 1000),
        }

        best_solution = bayesian_optimization(
            etr_function,
            parameters,
            n_iterations=cfg["MODELS"]["ML"]["BAYESIAN_ITERATIONS"],
        )

        params = best_solution["params"]

        model = ExtraTreesRegressor(
            n_estimators=int(max(params["n_estimators"], 0)),
            max_depth=int(max(params["max_depth"], 1)),
            min_samples_split=int(max(params["min_samples_split"], 2)),
            max_features=int(max(params["max_features"], 1)),
            n_jobs=-1,
            random_state=42,
        )

        return model, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "GradientBoosting":

        def gbr_function(  # pylint: disable=too-many-arguments
            learning_rate,
            min_samples_leaf,
            subsample,
            min_samples_split,
            max_depth,
            max_features,
            n_estimators,
        ):
            return cross_val_score(
                GradientBoostingRegressor(
                    loss="squared_error",
                    criterion="squared_error",
                    learning_rate=max(learning_rate, 1e-4),
                    min_samples_leaf=int(max(min_samples_leaf, 1)),
                    subsample=max(subsample, 0.1),
                    min_samples_split=int(max(min_samples_split, 2)),
                    max_depth=int(max(max_depth, 1)),
                    max_features=int(max(max_features, 1)),
                    n_estimators=int(max(n_estimators, 0)),
                    random_state=42,
                ),
                X=x_train,
                y=y_train,
                cv=cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            ).mean()

        parameters = {
            "learning_rate": (1e-4, 0.5),
            "min_samples_leaf": (1, 10),
            "subsample": (0.1, 0.9),
            "min_samples_split": (2, 10),
            "max_depth": (1, 300),
            "max_features": (10, x_train.shape[1]),
            "n_estimators": (10, 1000),
        }

        best_solution = bayesian_optimization(
            gbr_function,
            parameters,
            n_iterations=cfg["MODELS"]["ML"]["BAYESIAN_ITERATIONS"],
        )

        params = best_solution["params"]

        model = GradientBoostingRegressor(
            loss="squared_error",
            criterion="squared_error",
            learning_rate=(max(params["learning_rate"], 1e-4)),
            min_samples_leaf=int(max(params["min_samples_leaf"], 1)),
            subsample=max(params["subsample"], 0.1),
            min_samples_split=int(max(params["min_samples_split"], 2)),
            max_depth=int(max(params["max_depth"], 1)),
            max_features=int(max(params["max_features"], 10)),
            n_estimators=int(max(params["n_estimators"], 10)),
            random_state=42,
        )

        return model, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "NuSVR":

        def nusvr_function(C, gamma, nu):  # pylint: disable=invalid-name
            return cross_val_score(
                NuSVR(C=C, gamma=gamma, nu=nu, kernel="rbf"),
                X=x_train,
                y=y_train,
                cv=cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            ).mean()

        parameters = {"C": (0.1, 2), "gamma": (0.001, 0.1), "nu": (0.1, 0.9)}

        best_solution = bayesian_optimization(
            nusvr_function,
            parameters,
            n_iterations=cfg["MODELS"]["ML"]["BAYESIAN_ITERATIONS"],
        )

        params = best_solution["params"]

        model = NuSVR(
            C=max(params["C"], 0.1),
            gamma=max(params["gamma"], 1e-4),
            nu=max(params["nu"], 0.1),
            kernel="rbf",
        )

        return model, params

    model = RandomForestRegressor(
        bootstrap=False,
        max_depth=22,
        max_features=50,
        min_samples_split=4,
        n_estimators=80,
        n_jobs=-1,
    )
    params = {
        "max_depth": 22,
        "max_features": 50,
        "min_samples_split": 4,
        "n_estimators": 80,
        "bootstrap": False,
        "n_jobs": 1,
    }
    return model, params


def retrieve_id(cfg):
    """Retrieve the ID column of the test file

    Args:
        cfg (dict): configuration file

    Returns:
        numpy.array: IDs of the samples
    """
    data_files = os.listdir(os.path.join(cfg["DATA_DIR"], "test/"))

    for datafile in data_files:
        if "input" in datafile:
            test_data = pd.read_csv(
                os.path.join("../data/test", datafile), delimiter=",", decimal="."
            )

    return test_data["_ID"].to_numpy()


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def plot_feature_importance(importance, names, model_type):
    """Plot feature importance based on ML mpdel used

    Args:
        importance (list): feature importance values
        names (list): feature names
        model_type (string): Model used
    """

    names = ["feature" + str(name) for name in names]
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    fig = plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"][0:25], y=fi_df["feature_names"][0:25])
    # Add chart labels
    plt.title(model_type + " FEATURE IMPORTANCE (TOP 25)")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")

    top_logdir = "../features_description/"
    save_dir = generate_unique_logpath(top_logdir, model_type.lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(save_dir + "/feature_importance.png", dpi=500)
    plt.close(fig)
