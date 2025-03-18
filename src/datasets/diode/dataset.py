import os
import numpy as np
from PIL import Image
from typing import Dict, List
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('diode')
class DIODEDataset(BaseDataset):
    """DIODE depth dataset."""
    
    def __init__(self, config_path: str, split: str = 'val', batch_size: int = 1):
        """Initialize DIODE dataset."""
        super().__init__(config_path, split, batch_size)
        self.min_depth = 0.6
        self.max_depth = 350.0

    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset."""
        data_pairs = []

        for env_type in ["indoors", "outdoors"]:
            env_path = os.path.join(self.root_dir, env_type)
            if not os.path.exists(env_path):
                continue

            for scene_dir in sorted(os.listdir(env_path)):
                scene_path = os.path.join(env_path, scene_dir)
                if not os.path.isdir(scene_path):
                    continue

                for scan_dir in sorted(os.listdir(scene_path)):
                    scan_path = os.path.join(scene_path, scan_dir)
                    if not os.path.isdir(scan_path):
                        continue

                    for filename in sorted(os.listdir(scan_path)):
                        if not filename.endswith(".png") or "_depth_mask" in filename:
                            continue

                        base_name = filename[:-4]
                        rgb_path = os.path.join(scan_path, f"{base_name}.png")
                        depth_path = os.path.join(scan_path, f"{base_name}_depth.npy")
                        mask_path = os.path.join(
                            scan_path, f"{base_name}_depth_mask.npy"
                        )

                        if (
                            os.path.exists(rgb_path)
                            and os.path.exists(depth_path)
                            and os.path.exists(mask_path)
                        ):
                            data_pairs.append(
                                {
                                    "rgb": rgb_path,
                                    "depth": depth_path,
                                    "mask": mask_path,
                                }
                            )

        return sorted(data_pairs, key=lambda x: x["rgb"])

    def _load_depth(self, path: str) -> Image.Image:
        """Load DIODE depth map from .npy file and convert to PIL Image.
        
        Args:
            path: Path to depth map .npy file
            
        Returns:
            Depth map as PIL Image
        """
        try:
            # Load depth map as numpy array
            depth_np = np.load(path).squeeze()
            
            # Normalize for better visualization if needed
            # (This is optional and depends on how you want to visualize the depth)
            depth_normalized = depth_np.copy()
            if np.max(depth_normalized) > 0:
                depth_normalized = depth_normalized / np.max(depth_normalized) * 255
            
            # Convert to PIL Image (using mode 'F' for floating point)
            depth_img = Image.fromarray(depth_normalized.astype(np.float32), mode='F')
            
            return depth_img
            
        except Exception as e:
            print(f"Error loading depth from {path}: {str(e)}")
            # Return a blank image in case of error
            return Image.fromarray(np.zeros((768, 1024), dtype=np.float32), mode='F')
    
    def _get_valid_mask(self, depth: Image.Image) -> np.ndarray:
        """Get valid mask for DIODE depth map.
        
        Args:
            depth: Depth map as PIL Image
            
        Returns:
            Valid mask as numpy array (boolean)
        """
        # First convert the PIL Image to numpy array
        depth_np = np.array(depth)
        
        # Check if this is a path-based call (from __getitem__)
        if isinstance(depth_np, np.ndarray) and hasattr(depth, 'filename'):
            # We're working with the actual image, so we need to load the mask file
            mask_path = depth.filename.replace('_depth.npy', '_depth_mask.npy')
            if os.path.exists(mask_path):
                valid_mask = np.load(mask_path).squeeze()
                return valid_mask
        
        # Fallback to general valid range check
        valid_mask = np.logical_and(
            (depth_np > self.min_depth), (depth_np < self.max_depth)
        )
        
        return valid_mask
    
    def __getitem__(self, idx: int) -> Dict[str, Image.Image]:
        """Get a sample from the DIODE dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing RGB and depth as PIL Images
        """
        sample = self.data_pairs[idx]
        
        # Load RGB image
        rgb = self._load_rgb_image(sample['rgb'])
        
        # Load depth map
        depth = self._load_depth(sample['depth'])
        
        # Load mask directly from the mask file
        mask_path = sample['mask']
        mask_np = np.load(mask_path).squeeze()
        mask_pil = Image.fromarray(np.uint8(mask_np) * 255, mode='L')
        
        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask_pil,
            'rgb_path': sample['rgb'],
            'depth_path': sample['depth']
        }
        
    def _download(self):
        """Download and extract DIODE dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("DIODE dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_diode_genpercept.tar.gz?download=true"
        print("Downloading DIODE dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir),
        )