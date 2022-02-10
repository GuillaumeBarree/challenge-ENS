"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import pickle
import sys
import yaml

import numpy as np
import pandas as pd

import data.loader as loader
from tools.utils import load_model, retrieve_id


def inference_ml(cfg, model_path):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
    """

    # Load test data
    _, preprocessed_test_data = loader.main(cfg=cfg)

    # Test
    x_test = preprocessed_test_data["x_test"]

    # Load model
    model = pickle.load(open(model_path, "rb"))

    # Make predictions
    y_pred_test = model.predict(x_test)

    return y_pred_test.reshape(-1, 1)


def model_average(cfg):
    """Compute the average prediction

    Args:
        cfg (dict): configuration
    """

    if not cfg["TEST"]["AVERAGE"]["ACTIVE"]:
        print("You should use inference.py !")
        sys.exit()

    # Get sample IDs
    idx = retrieve_id(cfg=cfg)

    # Compute probabilities for every models
    models_predictions = []
    for _, elem in enumerate(cfg["TEST"]["AVERAGE"]["PATH"]):
        with open(elem["CONFIG"], "r") as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.Loader)

        models_predictions.append(
            inference_ml(cfg=config_file, model_path=elem["MODEL"])
        )

    # Compute mean prediction
    predictions = np.mean(np.concatenate(models_predictions, axis=1), axis=1)

    # Output format
    results = np.concatenate((idx.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)

    # Save csv file
    pd.DataFrame(results, columns=["_ID", "Y"]).astype({"_ID": "int32"}).to_csv(
        cfg["TEST"]["PATH_TO_CSV"], index=False
    )


if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Run model average
    model_average(cfg=config_file)
