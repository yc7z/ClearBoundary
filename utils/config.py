from pathlib import Path
from typing import Optional
import torch

class Config:
    def __init__(
        self,
        train_data_dir: str = ".",
        val_data_dir: str = ".",
        test_data_dir: str = ".",
        output_dir: str = ".",
        checkpoint_dir: str = ".",
        patch_size: int = 15,
        batch_size: int = 32,
        n_blocks: int = 6,
        n_heads: int = 4,
        d_model: int = 384,
        d_hidden: int = 384,
        dropout_p: float = 0.1,
        lr: float = 1e-4,
        num_epochs: int = 60,
        validate_every: int = 5,
        device: Optional[str] = None,
        mode: str = "train",
        load_from_checkpoint: bool = False,
    ):
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout_p = dropout_p
        self.lr = lr
        self.num_epochs = num_epochs
        self.validate_every = validate_every
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.load_from_checkpoint = load_from_checkpoint

        # Create directories if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Default checkpoint path
        self.checkpoint_path = self.checkpoint_dir / "best_model.pt"

    def __repr__(self):
        """
        Return a string representation of the configuration for easier debugging.
        """
        config_vars = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in config_vars.items())
