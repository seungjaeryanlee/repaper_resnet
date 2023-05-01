"""Train a model on image classification."""
import logging

import torch
import wandb
from omegaconf import OmegaConf

from configs import load_config
from data_loaders import init_data_loader
from datasets import init_dataset
from losses import init_loss
from lr_schedulers import init_lr_scheduler
from metrics import MetricsTracker
from models import init_model
from optimizers import init_optimizer
from samplers import init_sampler
from transforms import init_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.getLogger().setLevel(logging.INFO)


def train(CONFIG):
    """Train an image classification model."""
    if CONFIG.wandb.enable:
        wandb.login()
        wandb.init(
            entity=CONFIG.wandb.entity,
            project=CONFIG.wandb.project,
            name=CONFIG.wandb.name,
            config=OmegaConf.to_container(CONFIG),
        )

    ## Initialize components
    train_transform = init_transform(CONFIG.transforms.train)
    train_dataset = init_dataset(CONFIG.datasets.train, train_transform, train=True)
    train_sampler = init_sampler(CONFIG.samplers.train, train_dataset)
    train_loader = init_data_loader(
        train_dataset,
        train_sampler,
        CONFIG.data_loaders.train,
    )
    valid_transform = init_transform(CONFIG.transforms.valid)
    valid_dataset = init_dataset(CONFIG.datasets.valid, valid_transform, train=False)
    valid_sampler = init_sampler(CONFIG.samplers.valid, valid_dataset)
    valid_loader = init_data_loader(
        valid_dataset,
        valid_sampler,
        CONFIG.data_loaders.valid,
    )

    model = init_model(CONFIG.model).to(device)
    criterion = init_loss(CONFIG.loss)
    optimizer = init_optimizer(model.parameters(), CONFIG.optimizer)
    lr_scheduler = init_lr_scheduler(optimizer, CONFIG.lr_scheduler)

    ## Training loop
    for epoch_i in range(CONFIG.num_epochs):
        logging.info(f"Epoch {epoch_i}")

        ## Training Phase
        model.train()
        train_metrics = MetricsTracker()
        for minibatch_i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            losses.mean().backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_metrics.extend(losses=losses, labels=labels, preds=preds)

        ## Validation Phase
        model.eval()
        valid_metrics = MetricsTracker()
        for minibatch_i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                losses = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            valid_metrics.extend(losses=losses, labels=labels, preds=preds)

        ## Log each epoch
        if CONFIG.wandb.enable:
            wandb.log(
                data={
                    "epoch_i": epoch_i,
                    "lr": optimizer.param_groups[0]["lr"],
                    **train_metrics.get_metrics(prefix="train_"),
                    **valid_metrics.get_metrics(prefix="valid_"),
                },
                step=epoch_i,
            )

        # Update learning rate
        lr_scheduler.step()


if __name__ == "__main__":
    CONFIG = load_config()
    train(CONFIG)
