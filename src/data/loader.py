"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import argparse
import os
import torch
import yaml

from torch.utils.data import DataLoader
from preprocessing import apply_preprocessing
from dataset_utils import basic_random_split, RegressionDataset, load_test_data


def main(cfg):  # pylint: disable=too-many-locals
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """

    # Set path
    path_to_train = os.path.join(cfg["DATA_DIR"], "train/")
    path_to_test = os.path.join(cfg["DATA_DIR"], "test/")

    # Load the dataset for the training/validation sets
    data = basic_random_split(
        path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
    )
    preprocessed_data = apply_preprocessing(
        cfg=cfg["DATASET"]["PREPROCESSING"], data=data
    )

    # Load the test set
    test_data = load_test_data(path_to_test=path_to_test)
    preprocessed_test_data = apply_preprocessing(
        cfg=cfg["DATASET"]["PREPROCESSING"], data=test_data, test=True
    )

    # Train
    x_train = preprocessed_data["x_train"]
    y_train = preprocessed_data["y_train"]
    # Valid
    x_valid = preprocessed_data["x_valid"]
    y_valid = preprocessed_data["y_valid"]
    # Test
    x_test = preprocessed_test_data["x_test"]
    y_test = preprocessed_test_data["y_test"]

    # Create train, valid, test dataset
    train_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_train).float(),
        y_data=torch.from_numpy(y_train).float(),
    )
    valid_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_valid).float(),
        y_data=torch.from_numpy(y_valid).float(),
    )
    test_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_test).float(), y_data=torch.from_numpy(y_test).float()
    )

    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        num_workers=cfg["DATASET"]["NUM_THREADS"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg["TEST"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    if cfg["DATASET"]["VERBOSITY"]:
        print(
            f"The train set contains {len(train_loader.dataset)} samples,"
            f" in {len(train_loader)} batches"
        )
        print(
            f"The validation set contains {len(valid_loader.dataset)} samples,"
            f" in {len(valid_loader)} batches"
        )
        print(
            f"The test set contains {len(test_loader.dataset)} images,"
            f" in {len(test_loader)} batches"
        )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    main(cfg=config_file)
