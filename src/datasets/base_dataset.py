from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import yaml
import os
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import torchvision.transforms as transforms

class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize dataset.
        
        Args:
            config_path: Path to dataset config file
            split: Dataset split (train/val/test)
            batch_size: Batch size for loading data
        """
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set paths from config
        self.root_dir = self.config['paths']['root_dir']
        
        self._check_downloaded()
        
        self.split_dir = os.path.join(self.root_dir, self.config['paths'][split])
        
        # Set transforms
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['preprocessing']['rgb_mean'],
                std=self.config['preprocessing']['rgb_std']
            )
        ])
        
        # Load dataset structure
        self.data_pairs = self._traverse_directory()
        
    @abstractmethod
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Traverse directory and get pairs of RGB and depth paths.
        
        Returns:
            List of dicts containing paths for image and depth pairs
        """
        pass
        
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Image.Image]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Dict containing:
                - rgb: RGB image as PIL Image
                - depth: Depth map as PIL Image
                - mask: Valid depth mask as PIL Image
        """
        sample = self.data_pairs[idx]
        
        # Load RGB image as PIL
        rgb = self._load_rgb_image(sample['rgb'])
        
        # Load depth map as PIL
        depth = self._load_depth(sample['depth'])
        
        # Create valid mask as PIL
        mask = self._get_valid_mask(depth)
        mask_pil = Image.fromarray(np.uint8(mask) * 255)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask_pil,
            'rgb_path': sample['rgb'],
            'depth_path': sample['depth']
        }
    
    @abstractmethod
    def _load_depth(self, path: str) -> Image.Image:
        """Load depth map from file.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as PIL Image
        """
        depth = Image.open(path)
        return depth
    
    def _load_rgb_image(self, path: str) -> Image.Image:
        """Load RGB image from file.
        
        Args:
            path: Path to RGB image file
            
        Returns:
            RGB image as PIL Image
        """
        return Image.open(path).convert('RGB')
    
    def _get_valid_mask(self, depth: Image.Image) -> np.ndarray:
        """Get valid mask for depth map.
        
        Args:
            depth: Depth map as PIL Image
            
        Returns:
            Valid mask as numpy array (boolean)
        """
        depth_np = np.array(depth)
        valid_mask = np.logical_and(
            (depth_np > self.min_depth), (depth_np < self.max_depth)
        )
        return valid_mask
    
    @abstractmethod
    def _download(self):
        """Download dataset files. Must be implemented by child classes."""
        pass
    
    def _check_downloaded(self):
        """Ensures dataset is downloaded. Called before directory traversal."""
        if not os.path.exists(self.root_dir) or len(os.listdir(self.root_dir)) == 0:
            print(f"Dataset not found in {self.root_dir}, downloading...")
            self._download()
            
        if not os.path.exists(self.root_dir) or len(os.listdir(self.root_dir)) == 0:
            raise RuntimeError(f"Failed to download dataset to {self.root_dir}")
    
    def get_batch(self, start_idx: int) -> Tuple[Dict[str, List], int]:
        """Get a batch of samples.
        
        Args:
            start_idx: Starting index
            
        Returns:
            Batch of samples and next start index
        """
        batch_size = min(self.batch_size, len(self) - start_idx)
        batch = {
            'rgb': [],
            'depth': [],
            'mask': [],
            'rgb_path': [],
            'depth_path': []
        }
        
        for i in range(batch_size):
            sample = self[start_idx + i]
            for key in batch:
                batch[key].append(sample[key])
                
        return batch, start_idx + batch_size

# Dictionary to register dataset classes
DATASET_REGISTRY = {}

def register_dataset(name: str):
    """Decorator to register a new dataset class."""
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper