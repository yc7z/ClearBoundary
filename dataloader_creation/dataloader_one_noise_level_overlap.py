from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re

from utils import patches_fun


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
        img_pattern = re.compile(r'.*clean_img_boundaries.*')

        for image_folder in self.data_dir.iterdir():
            if not image_folder.is_dir() or 'lol' in str(image_folder):
                continue
            
            for file_name in image_folder.iterdir():
                if img_pattern.match(str(file_name)):
                    clean_image_path = file_name
                    
            if not clean_image_path.exists():
                raise FileNotFoundError(f"Missing clean image: {clean_image_path}")

            for noise_level_folder in image_folder.iterdir():
                if not noise_level_folder.is_dir() or '0.3' not in str(noise_level_folder):
                    continue

                noisy_image_paths = sorted(noise_level_folder.glob("*.png"))
                if not noisy_image_paths:
                    raise FileNotFoundError(f"No noisy images in {noise_level_folder}")

                data_points.append({
                    "clean": clean_image_path,
                    "noisy": noisy_image_paths,
                })

        return data_points

    def _extract_patches(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Splits an image into patches of size patch_size x patch_size."""
        _, h, w = image_tensor.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")
        
        patches = patches_fun.extract_patches_2ds(torch.unsqueeze(image_tensor, 0), self.patch_size, padding=5, stride=10)
        return [patch for patch in patches]

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_point = self.data_points[idx]
        clean_image = Image.open(data_point["clean"]).resize((150, 150))
        clean_image = transforms.ToTensor()(clean_image)
        noisy_images = [transforms.ToTensor()(Image.open(p).resize((150, 150))) for p in data_point["noisy"]]

        # Extract patches from clean and noisy images
        clean_patches = self._extract_patches(clean_image * (clean_image > 0.15))
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
