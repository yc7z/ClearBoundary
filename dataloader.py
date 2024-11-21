from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        patch_size: int,
        transform: callable = None,
    ):
        """
        Args:
            data_dir (Path): Path to the dataset directory.
            patch_size (int): Size of the patches to extract from each image.
            transform (callable): Transformations to apply to the patches.
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.transform = transform
        self.data_points = self._prepare_data()

    def _prepare_data(self) -> List[Dict[str, List[Path]]]:
        """Prepares the dataset by indexing clean and noisy images."""
        data_points = []

        for image_folder in self.data_dir.iterdir():
            if not image_folder.is_dir():
                continue

            clean_image_path = image_folder / "clean_img_boundaries.png"
            if not clean_image_path.exists():
                raise FileNotFoundError(f"Missing clean image: {clean_image_path}")

            for noise_level_folder in image_folder.iterdir():
                if not noise_level_folder.is_dir():
                    continue

                noisy_image_paths = sorted(noise_level_folder.glob("*.png"))
                if not noisy_image_paths:
                    raise FileNotFoundError(f"No noisy images in {noise_level_folder}")

                data_points.append({
                    "clean": clean_image_path,
                    "noisy": noisy_image_paths,
                })

        return data_points

    def _extract_patches(self, image: Image.Image) -> List[torch.Tensor]:
        """Splits an image into patches of size patch_size x patch_size."""
        image_tensor = transforms.ToTensor()(image)  # Convert to tensor
        _, h, w = image_tensor.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        patches = image_tensor.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)  # (num_patches, channels, patch_size, patch_size)
        return [patch for patch in patches]

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_point = self.data_points[idx]
        clean_image = Image.open(data_point["clean"]).convert("RGB")
        noisy_images = [Image.open(p).convert("RGB") for p in data_point["noisy"]]

        # Extract patches from clean and noisy images
        clean_patches = self._extract_patches(clean_image)
        noisy_patches_list = [self._extract_patches(noisy) for noisy in noisy_images]

        # Create data points for each patch
        inputs = []
        targets = []
        for patch_idx in range(len(clean_patches)):
            noisy_patches = [noisy_patches_list[ni][patch_idx] for ni in range(len(noisy_images))]
            inputs.append(torch.stack(noisy_patches))  # Shape: (num_noisy_patches, channels, patch_size, patch_size)
            targets.append(clean_patches[patch_idx])

        # Apply transformations
        if self.transform:
            inputs = [self.transform(inp) for inp in inputs]
            targets = [self.transform(tgt) for tgt in targets]

        return torch.stack(inputs), torch.stack(targets)


# if __name__ == '__main__':
#     data_dir = Path("data/")
#     patch_size = 15
#     transform = transforms.Compose([
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#     dataset = ImageDataset(data_dir=data_dir, patch_size=patch_size, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     for batch_inputs, batch_targets in dataloader:
#         print(f"Inputs: {batch_inputs.shape}, Targets: {batch_targets.shape}")
#         break
