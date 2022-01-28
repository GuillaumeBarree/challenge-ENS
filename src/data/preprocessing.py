"""This file contains all functions related to preprocessing."""
# pylint: disable=import-error
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def apply_preprocessing(cfg, data, test=False):
    """Normalize the data

    Args:
        cfg (dict): Preprocessing config dict
        data (dict:) train, valid and test inputs and targets.

    Returns:
        dict: Normalized data
    """
    # Preprocessing parameters
    type_ = cfg["NORMALIZE"]["TYPE"]
    n_components = cfg["PCA"]["N_COMPONENTS"]
    name = f"pipeline_{type_}_ncompo_{n_components}.pkl"

    if not test:
        scaler = MinMaxScaler() if type_ == "MinMaxScalar" else StandardScaler()
        pca = PCA(n_components=n_components)

        pipeline = Pipeline([("scaling", scaler), ("pca", pca)])

        data["x_train"] = pipeline.fit_transform(data["x_train"])
        data["x_valid"] = pipeline.transform(data["x_valid"])

        # Save the pipeline
        pickle.dump(pipeline, open(os.path.join("./data/normalized_data", name), "wb"))

        return data

    pipeline = pickle.load(open(os.path.join("./data/normalized_data", name), "rb"))
    data["x_test"] = pipeline.transform(data["x_test"])

    return data
