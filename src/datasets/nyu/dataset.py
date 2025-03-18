import os
import numpy as np
from PIL import Image
from typing import Dict, List

from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('nyu')
class NYUDataset(BaseDataset):
    """NYU Depth V2 dataset."""
    min_depth = 1e-3
    max_depth = 10.0
    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Traverse NYU dataset directory structure."""
        data_pairs = []
        split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.exists(split_dir):
            print(f"Split directory does not exist: {split_dir}")
            return data_pairs

        for scene in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            rgb_files = [
                f for f in os.listdir(scene_dir)
                if f.startswith("rgb_") and f.endswith(".png")
            ]

            for rgb_file in rgb_files:
                img_id = rgb_file.replace("rgb_", "").replace(".png", "")
                depth_file = f"depth_{img_id}.png"

                rgb_path = os.path.join(scene_dir, rgb_file)
                depth_path = os.path.join(scene_dir, depth_file)

                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    data_pairs.append({"rgb": rgb_path, "depth": depth_path})

        return sorted(data_pairs, key=lambda x: x["rgb"])

    def _load_depth(self, path: str) -> Image.Image:
        """Load NYU depth map.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as PIL Image
        """
        # NYU depths are stored in millimeters
        depth_img = Image.open(path)
        # We keep it as a PIL Image, but we'll handle the conversion during evaluation
        return depth_img
    
    def _get_valid_mask(self, depth: Image.Image) -> np.ndarray:
        """Get valid mask for depth map with NYU-specific crop.
        
        Args:
            depth: Depth map as PIL Image
            
        Returns:
            Valid mask as numpy array (boolean)
        """
        depth_np = np.array(depth).astype(np.float32) / 1000.0  # Convert mm to meters
        
        # Create base valid mask
        valid_mask = np.logical_and(
            (depth_np > self.min_depth), (depth_np < self.max_depth)
        )
        
        # Add NYU evaluation crop
        h, w = valid_mask.shape
        eval_mask = np.zeros_like(valid_mask, dtype=bool)
        eval_mask[45:471, 41:601] = True
        
        # Combine masks
        valid_mask = np.logical_and(valid_mask, eval_mask)
        
        return valid_mask
    
    def _download(self):
        """Download and extract NYU dataset if not already present."""
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("NYU dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_nyu_genpercept.tar.gz?download=true"
        print("Downloading NYU dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir)
        )