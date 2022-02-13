"""This module aims to launch a training procedure."""
# pylint: disable=import-error, no-name-in-module
import os
import argparse
from shutil import copyfile
import json
import pickle
import yaml
import numpy as np

from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from tools.trainer import train_one_epoch
from tools.utils import (
    load_model,
    launch_grid_search,
    launch_bayesian_opt,
    generate_unique_logpath,
    plot_feature_importance,
)
from tools.valid import test_one_epoch, ModelCheckpoint
import data.loader as loader


def main_ml(cfg, path_to_config):  # pylint: disable=too-many-locals
    """Main pipeline to train a ML model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """
    # Load data
    preprocessed_data, _ = loader.main(cfg=cfg)

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODELS"]["ML"]["TYPE"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    if cfg["MODELS"]["ML"]["GRID_SEARCH"]:
        model, params = launch_grid_search(cfg, preprocessed_data)

        with open(os.path.join(save_dir, "best_params.json"), "w") as outfile:
            json.dump(params, outfile, indent=2)

    elif cfg["MODELS"]["ML"]["BAYESIAN_OPT"]:
        model, params = launch_bayesian_opt(cfg, preprocessed_data)

        print(params)

        with open(os.path.join(save_dir, "best_params.json"), "w") as outfile:
            json.dump(params, outfile, indent=2)

    else:
        model = load_model(cfg=cfg)

    model.fit(X=preprocessed_data["x_train"], y=preprocessed_data["y_train"])
    pickle.dump(model, open(os.path.join(save_dir, "model.pck"), "wb"))

    y_pred = model.predict(preprocessed_data["x_valid"])

    print("Valid MSE : ", mean_squared_error(preprocessed_data["y_valid"], y_pred))
    print(
        "Train MSE : ",
        mean_squared_error(
            preprocessed_data["y_train"], model.predict(preprocessed_data["x_train"])
        ),
    )

    if cfg["MODELS"]["ML"]["TYPE"] in [
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
    ]:
        plot_feature_importance(
            model.feature_importances_,
            (np.arange(0, preprocessed_data["x_train"].shape[1])),
            cfg["MODELS"]["ML"]["TYPE"],
        )


def main_nn(cfg, path_to_config):  # pylint: disable=too-many-locals
    """Main pipeline to train a NN model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """

    # Load data
    train_loader, valid_loader, _ = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    input_size = train_loader.dataset[0][0].shape[0]

    model = load_model(cfg=cfg, input_size=input_size)
    model = model.to(device)

    # Define the loss
    f_loss = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["TRAIN"]["LR_INITIAL"])

    # Tracking with tensorboard
    tensorboard_writer = SummaryWriter(log_dir=cfg["TRAIN"]["LOG_DIR"])

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["TRAIN"]["MODEL"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    # Init Checkpoint class
    checkpoint = ModelCheckpoint(
        save_dir, model, cfg["TRAIN"]["EPOCH"], cfg["TRAIN"]["CHECKPOINT_STEP"]
    )

    # Lr scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["TRAIN"]["LR_DECAY"],
        patience=cfg["TRAIN"]["LR_PATIENCE"],
        threshold=cfg["TRAIN"]["LR_THRESHOLD"],
    )

    # Launch training loop
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH : {}".format(epoch))

        train_loss, train_mse = train_one_epoch(
            model, train_loader, f_loss, optimizer, device
        )
        val_loss, val_mse = test_one_epoch(model, valid_loader, f_loss, device)

        # Update learning rate
        scheduler.step(val_loss)
        learning_rate = scheduler.optimizer.param_groups[0]["lr"]

        # Save best checkpoint
        checkpoint.update(val_loss, epoch)

        # Track performances with tensorboard
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_loss"), train_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_mse"), train_mse, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_loss"), val_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_mse"), val_mse, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "lr"), learning_rate, epoch
        )


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

    if config_file["MODELS"]["NN"]:
        main_nn(cfg=config_file, path_to_config=args.path_to_config)

    else:
        main_ml(cfg=config_file, path_to_config=args.path_to_config)
