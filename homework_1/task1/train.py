"""
train.py

"""
# TODO imp via arg
# TODO implement callbacks best model checkpoint
# TODO implement num of epoch to be passed via arg

import torch
from torch import nn
from tqdm.auto import tqdm
from dataset import get_train_data
from unet import Unet
from loguru import logger
from scaler import LossScaler

logger.add("log_{time}.log")


def train_epoch(train_loader, model, criterion, optimizer, device, amp=False):
    """

    """
    scaler = LossScaler()
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (images, labels) in pbar:
        images - images.to(device)
        labels = labels.to(device)
        if amp is False:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.info("Calc accuracy")
            accuracy = ((outputs > 0.5) == labels).float().mean()
            pbar.set_description(
                f"Loss: {round(loss.item(), 4)}" f"Accuracy: {round(accuracy.item() * 100, 4)}")

        else:
            with torch.amp.autocast_mode(device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # TODO code for losss scalling (static and Dynamic loss scalling)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                


def train(num_epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    for epoch in range(0, num_epochs):
        logger.info(f"Training epoch {epoch} started")
        train_epoch(train_loader=train_loader, model=model,
                    criterion=criterion, device=device, optimizer=optim)


if __name__ == "__main__":
    # TODO add nums_epoch as command arg
    logger.info("Starting training process")
    train()
