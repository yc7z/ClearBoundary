import wandb
from dataloader import ImageDataset
from pathlib import Path
from torch.utils.data import DataLoader
from train import Trainer

def main(config):
    wandb.login()
    wandb.init(project="clarify_boundary_transformer", config=config)
    config = wandb.config

    # Dataset and DataLoader
    train_dataset = ImageDataset(
        data_dir=Path(config.train_data_dir),
        patch_size=config.patch_size,
    )
    val_dataset = ImageDataset(
        data_dir=Path(config.val_data_dir),
        patch_size=config.patch_size,
    )
    test_dataset = ImageDataset(
        data_dir=Path(config.test_data_dir),
        patch_size=config.patch_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Initialize Trainer
    trainer = Trainer(config)

    if config.mode == "train":
        trainer.train(train_loader, val_loader)
    elif config.mode == "evaluate":
        trainer.load_model(trainer.best_model_path)
        trainer.evaluate(test_loader, config.output_dir)


if __name__ == "__main__":
    # Example config
    config = {
        "train_data_dir": "path/to/train",
        "val_data_dir": "path/to/val",
        "test_data_dir": "path/to/test",
        "output_dir": "path/to/output",
        "checkpoint_dir": "path/to/checkpoints",
        "patch_size": 32,
        "batch_size": 16,
        "n_blocks": 4,
        "n_heads": 8,
        "d_model": 256,
        "d_hidden": 512,
        "dropout_p": 0.1,
        "lr": 1e-4,
        "num_epochs": 100,
        "validate_every": 5,
        "device": "cuda",
        "mode": "train",  # Can be "train" or "evaluate"
    }
    main(config)
