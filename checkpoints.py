import os

import torch


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    checkpoint_filepath: str,
    finished_epoch,
):
    checkpoint_dirpath = "/".join(checkpoint_filepath.split("/")[:-1])
    os.makedirs(checkpoint_dirpath, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "lr_scheduler_state_dict": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
            "finished_epoch": finished_epoch,
        },
        checkpoint_filepath,
    )


def load_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    checkpoint_filepath: str,
):
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start epoch is one after the checkpointed epoch, because we checkpoint after
    # finishing the epoch.
    return checkpoint["finished_epoch"] + 1
