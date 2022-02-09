"""This file contains all functions related to the dataset."""
# pylint: disable=import-error
import os
import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """Create a Torch Dataset for our regression problem."""

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)


def basic_random_split(path_to_train, valid_ratio=0.2):
    """This function split file according to a ratio to create
    training and validation.

    Args:
        path_to_train (str): path of the data root directory.
        valid_ratio (float): ratio of data for validation dataset.

    Returns:
        dict: Dictionary containing every data to create a Dataset.
    """
    # Load the different files
    training_data = load_files(path_to_data=path_to_train)

    # Prepare features and targets
    features_and_targets = remove_useless_features(training_data=training_data)

    features_and_targets = create_x_and_y(
        input_data=features_and_targets, valid_ratio=valid_ratio
    )

    return features_and_targets


def load_test_data(path_to_test):
    """This function load test data

    Args:
        path_to_test (str): path of the data root directory.

    Returns:
        dict: Dictionary containing every data to create a Dataset.
    """

    # Load the different files
    test_data = load_files(path_to_data=path_to_test)

    # Drop useless
    test_data["input"] = test_data["input"].drop(columns=["_ID"])

    # Create a target
    test_data["target"] = np.ones((len(test_data["input"])))

    feature_and_target = {
        "x_test": test_data["input"].to_numpy(),
        "y_test": np.ones((len(test_data["input"]))).ravel(),
    }

    return feature_and_target


def load_files(path_to_data):
    """Load data input files.

    Args:
        path_to_data (str): path of the data root directory.

    Returns:
        list(pandas.core.frame.DataFrame): List of Dataframe containing data from each file.
    """
    data = {}
    data_files = os.listdir(path_to_data)

    for datafile in tqdm.tqdm(data_files):
        if "input" in datafile:
            data["input"] = pd.read_csv(
                os.path.join(path_to_data, datafile), delimiter=",", decimal="."
            )
        else:
            data["target"] = pd.read_csv(
                os.path.join(path_to_data, datafile), delimiter=",", decimal="."
            )
    return data


def remove_useless_features(training_data):
    """Create features and targets

    Args:
        training_data (list): List of Dataframe containing data from each file.

    Returns:
        dict : Dictionary containing features and target for each file.
    """
    data_dict = {}

    for key, data in training_data.items():

        features = data.drop(columns=["_ID"])
        data_dict[key] = features

    return data_dict


def create_x_and_y(input_data, valid_ratio):  # pylint: disable=too-many-locals
    """Generate train, valid and test for each file and for each target.

    Args:
        input_data (dict): Features and targets for one file.
        valid_ratio (float): Test and validation ratio.

    Returns:
        dict: train, valid and test inputs and targets.
    """
    feature_and_target = {}

    x_train, x_valid, y_train, y_valid = train_test_split(
        input_data["input"], input_data["target"], test_size=valid_ratio, random_state=0
    )
    y_train = y_train.values.ravel()
    y_valid = y_valid.values.ravel()

    feature_and_target = {
        "x_train": x_train.to_numpy(),
        "y_train": y_train,
        "x_valid": x_valid.to_numpy(),
        "y_valid": y_valid,
    }

    return feature_and_target
