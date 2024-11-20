import torch

def save_checkpoint(model, epoch, checkpoint_path):
    """
    Save model state and epoch number to a checkpoint file.

    Args:
        model: The model to save.
        epoch: The current epoch number.
        checkpoint_path: Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model state, optimizer state, and epoch number from a checkpoint file.

    Args:
        model: The model to load the state into..
        device: The device to map the loaded state.

    Returns:
        int: The epoch number from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {epoch}")
    return epoch
