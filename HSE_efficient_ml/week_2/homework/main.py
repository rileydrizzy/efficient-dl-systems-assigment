import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from loguru import logger
from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on {device}")
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(config=config_dict, project="effdl_week_2")
        # Log the run
        wandb.run.log_code(name="Hydra_Config", root="./config/config.yaml")

        ddpm = DiffusionModel(
            eps_model=UnetModel(
                cfg.model_params.UnetModel.in_channel,
                cfg.model_params.UnetModel.out_channel,
                cfg.model_params.UnetModel.hidden_size,
            ),
            betas=cfg.model_params.DiffusionModel.betas,
            num_timesteps=cfg.model_params.DiffusionModel.num_timesteps,
        )
        ddpm.to(device)
        train_transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if cfg.params.random_flip:
            train_transforms_list.append(transforms.RandomHorizontalFlip())
        train_transforms = transforms.Compose(train_transforms_list)

        dataset = CIFAR10(
            "cifar10",
            train=True,
            download=True,
            transform=train_transforms,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.params.batchsize,
            num_workers=cfg.params.num_workers,
            shuffle=True,
        )

        if cfg.optimizer.SGD == "SGD":
            optim = torch.optim.SGD(
                ddpm.parameters(),
                lr=cfg.optimizer.SGD.lr,
                momentum=cfg.optimizer.SGD.momentum,
            )
        else:
            optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.optimizer.Adam.lr)

        wandb.watch(ddpm)
        logger.info(f"Starting training for {cfg.params.num_epochs} Epochs")
        for i in range(cfg.params.num_epochs):
            epoch_loss = train_epoch(ddpm, dataloader, optim, device)
            generate_samples(ddpm, device, f"samples/{i:02d}.png")
        wandb.log({f"{i}_sample": f"samples/{i:02d}.png", f"{i}_epoch": epoch_loss})

        logger.info(f"Training the Unet Model finished, final loss-> {epoch_loss:.2f}")
    except Exception as error:
        logger.exception(f" Training failed due to {error}")


if __name__ == "__main__":
    main()
