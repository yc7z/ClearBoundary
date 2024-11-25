import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
import wandb
from pathlib import Path
from typing import Optional, Tuple
from torchvision import transforms

from models.transformer import Transformer
from utils.config import Config
from checkpointer import save_checkpoint, load_checkpoint
import math
import numpy as np
import os

class Trainer:
    def __init__(self, config: Config, device=None):
        self.config = config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_id = config.run_id
        self.criterion = MSELoss()
        self.best_val_loss = float("inf")
        self.best_model_path = Path(config.checkpoint_dir) / f"best_model_run{self.run_id}.pth"
        self.model = self.initialize_model(config.load_from_checkpoint).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)

    def initialize_model(self, load_from_checkpoint: bool = False):
        """Initialize the model."""
        model = Transformer(
            n_blocks=self.config.n_blocks,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            d_hidden=self.config.d_hidden,
            d_input=self.config.patch_size**2 * 1,
            d_output=self.config.patch_size**2 * 1,
            dropout_p=self.config.dropout_p,
        )
        if load_from_checkpoint:
            last_epoch = load_checkpoint(model, self.best_model_path, self.device)
            self.config.num_epochs -= last_epoch
        return model

    def train_one_epoch(self, dataloader: DataLoader):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for noisy_patches, clean_patches in dataloader:
            noisy_patches, clean_patches = self.prepare_data(noisy_patches, clean_patches)

            self.optimizer.zero_grad()
            outputs = self.model(noisy_patches)
            loss = self.criterion(outputs, clean_patches)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * noisy_patches.size(0)

        return epoch_loss / len(dataloader.dataset)

    def validate_one_epoch(self, dataloader: DataLoader):
        """Validate the model for one epoch."""
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for noisy_patches, clean_patches in dataloader:
                noisy_patches, clean_patches = self.prepare_data(noisy_patches, clean_patches)

                outputs = self.model(noisy_patches)
                loss = self.criterion(outputs, clean_patches)

                epoch_loss += loss.item() * noisy_patches.size(0)

        return epoch_loss / len(dataloader.dataset)

    def prepare_data(self, noisy_patches, clean_patches) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        noisy_patches = noisy_patches.to(self.device)
        clean_patches = clean_patches.to(self.device)

        # Flatten patches for the transformer
        noisy_patches = torch.squeeze(noisy_patches)
        clean_patches = torch.squeeze(clean_patches)
        noisy_patches = noisy_patches.squeeze().view(noisy_patches.size(0), noisy_patches.size(1), -1)
        # noisy_patches = noisy_patches.squeeze().view(noisy_patches.size(0), -1)
        clean_patches = clean_patches.squeeze().view(clean_patches.size(0), -1)
        # clean_patches = torch.unsqueeze(clean_patches, 0)
        # clean_patches = clean_patches.view(clean_patches.size(0), -1)
        # clean_patches = torch.squeeze(clean_patches)
        
        # print(clean_patches.shape)
        # print(noisy_patches.shape)

        return noisy_patches, clean_patches

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model."""
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            wandb.log({"epoch": epoch, "train_loss": train_loss})

            if (epoch + 1) % self.config.validate_every == 0:
                val_loss = self.validate_one_epoch(val_loader)
                wandb.log({"epoch": epoch, "val_loss": val_loss})

                # Save the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(self.model, epoch, self.best_model_path)
                    
                    artifact = wandb.Artifact("best_model_checkpoint", type="model")
                    artifact.add_file(str(self.best_model_path))
                    wandb.log_artifact(artifact)

    def evaluate(self, test_loader: DataLoader, output_dir: Optional[Path] = None) -> None:
        """Evaluate the model and optionally save outputs."""
        self.model.eval()
        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx, (noisy_patches, clean_patches) in enumerate(test_loader):
                noisy_patches, clean_patches = self.prepare_data(noisy_patches, clean_patches)

                # outputs has size (batch_size, num_patches, n_channels, patch_size, patch_size)
                outputs = self.model(noisy_patches)
                D = int(math.sqrt(outputs.size(0)) * self.config.patch_size)
                outputs = outputs.view(-1, 1, self.config.patch_size, self.config.patch_size)
                clean_patches = clean_patches.view(-1, 1, self.config.patch_size, self.config.patch_size)

                if not(os.path.exists(f'{output_dir}/input_imgs_{idx}') and os.path.isdir(f'{output_dir}/input_imgs_{idx}')):
                    os.mkdir(f'{output_dir}/input_imgs_{idx}')
 
                for i in range(noisy_patches.shape[1]):
                    input_reconstruction = reconstruct_image(patches=noisy_patches[:,i,:].view(-1, 1, self.config.patch_size, self.config.patch_size), original_shape=(D, D))
                    input_image = transforms.ToPILImage()(input_reconstruction.cpu())
                    input_image.save(f'{output_dir}/input_imgs_{idx}/test_{idx}_input_{i}.png')
                    
                if output_dir:
                    # for i in range(outputs.size(0)):
                    #     output_patch = transforms.ToPILImage()(outputs[i].cpu())
                    #     clean_patch = transforms.ToPILImage()(clean_patches[i].cpu())
                        
                    #     output_patch.save(output_dir / f"test_{idx}_output_patch_{i}.png")
                    #     clean_patch.save(output_dir / f"test_{idx}_clean_patch_{i}.png")
                    
                    # Save the entire image.
                    outputs_reconstruction = reconstruct_image(patches=outputs, original_shape=(D, D))
                    clean_patches_reconstruction = reconstruct_image(patches=clean_patches, original_shape=(D, D))
                    output_image = transforms.ToPILImage()(np.clip(outputs_reconstruction.cpu(), 0, 1))
                    clean_image = transforms.ToPILImage()(np.clip(clean_patches_reconstruction.cpu(), 0, 1))
                    output_image.save(output_dir / f"test_{idx}_output_image.png")
                    clean_image.save(output_dir / f"test_{idx}_clean_image.png")


def reconstruct_image(patches: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Reconstructs the original image from patches.

    Args:
        patches (torch.Tensor): Tensor of shape (n_patches, 3, patch_size, patch_size).
        original_shape (Tuple[int, int]): Original image dimensions (height, width).

    Returns:
        torch.Tensor: Reconstructed image of shape (3, original_height, original_width).
    """
    n_patches, channels, patch_size, _ = patches.shape
    original_height, original_width = original_shape

    # Check dimensions
    assert original_height % patch_size == 0 and original_width % patch_size == 0, \
        "Original image dimensions must be divisible by patch size."

    # Calculate grid size
    h_patches = original_height // patch_size
    w_patches = original_width // patch_size

    # Reshape patches into grid
    image = (
        patches.view(h_patches, w_patches, channels, patch_size, patch_size)
        .permute(2, 0, 3, 1, 4)  # Reorder dimensions to (C, H, h_patches, W, w_patches)
        .contiguous()
        .view(channels, original_height, original_width)
    )
    return image