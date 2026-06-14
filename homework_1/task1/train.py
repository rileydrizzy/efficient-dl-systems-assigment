"""
train.py
This module implements the training loop for a UNet model using PyTorch. 
It includes functionality for mixed precision training and loss scaling to prevent numerical instability. 

Functions:
- train_epoch: A function that performs one epoch of training, including forward pass, loss computation, backward pass, and optimization step. 
    It also calculates and logs the training loss and accuracy.
- train: A function that sets up the training environment, including device configuration, model initialization, loss function, optimizer and data loader.
    It then iterates through the specified number of epochs and calls the train_epoch function for each epoch.
- seed_everything: A function that sets the random seed for reproducibility across various libraries and configurations.
- get_parser: A function that defines and parses command-line arguments for the training process, 
    including the number of epochs, whether to use automatic mixed precision (AMP), whether to use loss scaling, and the mode of the loss scaler.

Usage:
To run the training process, execute the script with the desired command-line arguments. For example:
    `python train.py --epochs 10 --amp --loss_scaling --scaler-mode dynamic`

Author: Ladipo Ipadeola
Date: 06/14/2026        

"""

import argparse
import torch
from torch import nn
from tqdm.auto import tqdm
from dataset import get_train_data
from unet import Unet
from loguru import logger
from scaler import LossScaler

logger.add("log_{time}.log")


def train_epoch(train_loader, model, criterion, optimizer, device,  loss_scaling, amp, scaler=None):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if not amp:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if loss_scaling is True:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()
        accuracy = ((outputs > 0.5) == labels).float().mean()
        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logger.info(f" Running on {device} Machine")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()
    scaler_ = LossScaler(mode=args.scaler_mode)
    for epoch in range(0, args.epochs):
        logger.info(f"Training epoch {epoch} started")
        train_epoch(train_loader=train_loader, model=model,
                    criterion=criterion, device=device, optimizer=optim, scaler=scaler_, loss_scaling=args.loss_scaling, amp=args.amp)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--epochs", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("-s", "--loss_scaling", action="store_true")
    parser.add_argument("-sm", "--scaler-mode", type=str, default="static")

    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()
    args = get_parser()
    logger.info("Starting training process")
    logger.info(f"Total Training Epochs is {args.epochs}")
    if args.amp:
        logger.info("Training in Half Precision mode")
    else:
        logger.info("Training in Full Precision mode")
    if args.loss_scaling:
        logger.info(f"Scaler mode is {args.scaler_mode}")
    train(args)
    logger.info("Training completed successfully")
