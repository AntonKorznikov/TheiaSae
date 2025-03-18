import torch
from sae import BatchTopKSAE
from tensor_activation_store import TensorStorageActivationStore
from config import get_default_cfg, post_init_cfg
from training import train_sae
import os
import sys

# Add the parent directory to the path so we can import TensorStorage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.tensor_storage import TensorStorage

def main():
    # Get configuration
    cfg = get_default_cfg()
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = post_init_cfg(cfg)
    
    # Print key configuration settings
    print(f"Training SAE with the following configuration:")
    print(f"  - Tensor storage path: {cfg['tensor_storage_path']}")
    print(f"  - Dictionary size: {cfg['dict_size']}")
    print(f"  - SAE type: {cfg['sae_type']}")
    print(f"  - Top K: {cfg['top_k']}")
    print(f"  - Batch size: {cfg['batch_size']}")
    print(f"  - Learning rate: {cfg['lr']}")
    print(f"  - Device: {cfg['device']}")
    print(f"  - Input unit norm: {cfg['input_unit_norm']}")
    print(f"  - Shuffle: {cfg['shuffle']}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize sparse autoencoder
    print("Initializing sparse autoencoder...")
    sae = BatchTopKSAE(cfg)
    
    # Initialize tensor storage
    print(f"Loading tensor storage from {cfg['tensor_storage_path']}...")
    try:
        tensor_storage = TensorStorage(cfg['tensor_storage_path'])
        print(f"Loaded tensor storage with {len(tensor_storage)} embeddings")
        
        # Verify the embedding dimension matches the configured act_size
        sample_embedding = tensor_storage[0]
        if sample_embedding.shape[0] != cfg['act_size']:
            print(f"Warning: Embedding dimension ({sample_embedding.shape[0]}) does not match configured act_size ({cfg['act_size']})")
            print(f"Updating act_size to {sample_embedding.shape[0]}")
            cfg['act_size'] = sample_embedding.shape[0]
            # Reinitialize SAE with updated act_size
            sae = BatchTopKSAE(cfg)
    except Exception as e:
        print(f"Error loading tensor storage: {e}")
        return
    
    # Initialize activation store
    print("Initializing TensorStorageActivationStore...")
    activations_store = TensorStorageActivationStore(tensor_storage, cfg)
    
    # Train the SAE
    print("Starting SAE training...")
    train_sae(sae, activations_store, cfg)
    print("Training complete!")

if __name__ == "__main__":
    main()