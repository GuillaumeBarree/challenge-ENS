"""This module aims to define utils function for the project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
            "max_depth": np.arange(18, 19, 1),
            "max_features": np.arange(x_train.shape[1] - 1, x_train.shape[1], 5),
            "n_estimators": np.arange(70, 120, 5),
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
            "max_depth": np.arange(18, 28, 1),
            "max_features": np.arange(50, x_train.shape[1], 5),
            "n_estimators": np.arange(70, 120, 5),
        }

        etr_cv = GridSearchCV(etr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        etr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in etr_cv.best_params_.items():
            params[key] = int(value)

        return etr_cv.best_estimator_, params

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
