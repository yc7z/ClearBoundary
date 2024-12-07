import wandb
from dataloader_three_channels import ImageDataset
from pathlib import Path
from torch.utils.data import DataLoader
from train_three_channels import Trainer
from utils.config import Config

def main(config: Config):
    wandb.login()
    wandb.init(project="clarify_boundary_transformer", config=config)

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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize Trainer
    trainer = Trainer(config)

    if config.mode == "train":
        trainer.train(train_loader, val_loader)
    elif config.mode == "evaluate":
        trainer.evaluate(test_loader, config.output_dir)


if __name__ == "__main__":
    train_config = Config(
        train_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_train",
        val_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_val",
        test_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_test",
        output_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/output_on_test_23",
        checkpoint_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/checkpoints",
        patch_size=8,
        n_blocks=6,
        n_heads=4,
        d_model=384,
        d_hidden=384,
        dropout_p=0.1,
        lr=2e-4,
        num_epochs=20,
        validate_every=1,
        mode="train",
        load_from_checkpoint=False,
        run_id=23,
    )
    main(train_config)
    
    test_config = Config(
        train_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_train",
        val_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_val",
        test_data_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/noisy_imagenet64_test",
        output_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/output_on_test_23",
        checkpoint_dir="/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/checkpoints",
        patch_size=8,
        n_blocks=6,
        n_heads=4,
        d_model=384,
        d_hidden=384,
        dropout_p=0.1,
        lr=2e-4,
        num_epochs=1,
        validate_every=1,
        mode="evaluate",
        load_from_checkpoint=True,
        run_id=23,
    )
    
    main(test_config)