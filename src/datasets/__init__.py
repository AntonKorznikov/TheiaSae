from .base_dataset import BaseDataset, DATASET_REGISTRY
from .nyu import NYUDataset
from .kitti import KITTIDataset
from .diode import DIODEDataset

def build_dataset(name: str, config_path: str, **kwargs):
    """Build a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
        
    return DATASET_REGISTRY[name](config_path, **kwargs)

__all__ = [
    'BaseDataset',
    'NYUDataset',
    'build_dataset',
    'DATASET_REGISTRY',
    'KITTIDataset',
    'DIODEDataset',
]