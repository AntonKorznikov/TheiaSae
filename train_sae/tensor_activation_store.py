import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class TensorStorageActivationStore:
    """
    Activation store for embeddings pre-computed and stored in TensorStorage.
    Loads embeddings in batches from the TensorStorage and provides them for SAE training.
    """
    def __init__(self, tensor_storage, cfg: dict):
        """
        Initialize a TensorStorageActivationStore.
        
        Args:
            tensor_storage: The TensorStorage instance containing embeddings
            cfg: Configuration dictionary
        """
        self.tensor_storage = tensor_storage
        self.config = cfg
        self.batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.dtype = cfg["dtype"]
        self.shuffle = cfg.get("shuffle", False)  # Default to False as requested
        self.current_idx = 0
        self.total_samples = len(tensor_storage)
        
        # Create a sequential index array
        self.indices = list(range(self.total_samples))
        
        print(f"Initialized TensorStorageActivationStore with {self.total_samples} embeddings")
        print(f"Shuffle mode: {self.shuffle}")
        
        # Initialize buffer, dataloader, and iterator right away
        print("Initializing first batch of data...")
        self.activation_buffer = self._fill_buffer()
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _fill_buffer(self):
        """
        Fill the activation buffer with a batch of embeddings from tensor storage.
        """
        # Determine how many samples to process
        num_samples = min(
            self.config["num_batches_in_buffer"] * self.batch_size,
            self.total_samples - self.current_idx
        )
        
        if num_samples <= 0:
            # Reset the index if we've gone through all samples
            print("Resetting index to beginning of dataset")
            self.current_idx = 0
            num_samples = min(
                self.config["num_batches_in_buffer"] * self.batch_size,
                self.total_samples
            )
        
        all_activations = []
        end_idx = self.current_idx + num_samples
        
        print(f"Loading embeddings from index {self.current_idx} to {end_idx-1}")
        for i in tqdm(range(self.current_idx, end_idx), desc="Loading embeddings from storage"):
            idx = self.indices[i]
            # Get embedding and convert to PyTorch tensor
            embedding = torch.tensor(
                self.tensor_storage[idx], 
                dtype=self.dtype, 
                device=self.device
            )
            all_activations.append(embedding)
            
        self.current_idx = end_idx
        print(f"Loaded {len(all_activations)} embeddings")
        
        # Stack all embeddings into a single tensor
        stacked_tensor = torch.stack(all_activations, dim=0)
        print(f"Stacked tensor shape: {stacked_tensor.shape}")
        return stacked_tensor

    def _get_dataloader(self):
        """
        Create a PyTorch DataLoader over the activation buffer.
        """
        dataset = TensorDataset(self.activation_buffer)
        print(f"Created dataset with {len(dataset)} samples")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle  # Using the shuffle setting from config
        )

    def next_batch(self):
        """
        Return the next mini-batch of activations.
        """
        try:
            # Try to get the next batch from the current iterator
            batch = next(self.dataloader_iter)[0]
            return batch
        except (StopIteration, AttributeError) as e:
            print(f"Refreshing data buffer: {type(e).__name__}")
            
            # Clean up to free memory
            del self.activation_buffer
            del self.dataloader
            del self.dataloader_iter
            
            # Reload buffer with new data
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            
            # Get first batch from new iterator
            return next(self.dataloader_iter)[0]