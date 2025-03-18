import os
import logging
import torch
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Iterator, Any
import numpy as np
import argparse
import json
import shutil
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Import TensorStorage from the provided code
from src.datasets.tensor_storage import TensorStorage

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
TARGET_SIZE = (256, 256)
DATASET_PATH = "data/imagenet_1k_resized_256"

def extract_depth_anything_embeddings(
    model_variant: str,
    dataset,
    storage_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
    debug: bool = False,
    debug_samples: int = 1000,
) -> TensorStorage:
    """Extract embeddings from Depth Anything model and store them.
    
    Args:
        model_variant: Model variant (small, base, large)
        dataset: The dataset to process
        storage_dir: Directory to save embeddings
        batch_size: Batch size for processing
        device: Device to use for computation (cuda or cpu)
        debug: If True, process only limited number of samples
        debug_samples: Number of samples to process in debug mode
        
    Returns:
        TensorStorage: Storage containing embeddings and metadata
    """
    logger.info(f"Initializing Depth Anything {model_variant} model...")
    
    # Load the model and processor
    model_name = f"LiheYoung/depth-anything-{model_variant}-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate total samples
    if debug:
        total_samples = min(debug_samples, len(dataset))
        logger.info(f"Running in debug mode with {total_samples} samples")
    else:
        total_samples = len(dataset)
    
    # Process the dataset and save embeddings and metadata to temporary files
    temp_embeddings_dir = os.path.join(storage_dir, "temp_embeddings")
    temp_metadata_dir = os.path.join(storage_dir, "temp_metadata")
    os.makedirs(temp_embeddings_dir, exist_ok=True)
    os.makedirs(temp_metadata_dir, exist_ok=True)
    
    processed_samples = 0
    total_valid_samples = 0
    
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx, batch in enumerate(tqdm(dataset.iter(batch_size=batch_size), 
                                          total=total_batches, 
                                          desc=f"Processing depth-anything-{model_variant}")):
        if processed_samples >= total_samples:
            break
            
        # Get batch of images
        batch_images = batch['image']
        batch_labels = batch.get('label', [-1] * len(batch_images))
        batch_class_names = batch.get('class_name', ['unknown'] * len(batch_images))
        
        # Preprocess images
        processed_images = []
        valid_indices = []  # Track which images processed successfully
        
        for idx, img in enumerate(batch_images):
            if processed_samples + idx >= total_samples:
                break
            
            # Convert to PIL if it's not already
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # Resize to target size
            img = img.resize(TARGET_SIZE)
            processed_images.append(img)
            valid_indices.append(idx)
        
        if not processed_images:
            processed_samples += len(batch_images)
            continue
            
        # Process with the model's processor - process in smaller groups to avoid memory issues
        # Process 8 images at a time to avoid memory issues
        sub_batch_size = 8
        all_embeddings = []
        
        for i in range(0, len(processed_images), sub_batch_size):
            sub_batch = processed_images[i:i+sub_batch_size]
            inputs = processor(images=sub_batch, return_tensors="pt").to(device)
        
            # Extract embeddings from the backbone (DINOv2 encoder part)
            with torch.no_grad():
                # The backbone output doesn't have last_hidden_state attribute
                # Instead, we need to get the hidden states directly from the backbone's encoder output
                # and apply the layernorm
                
                # First get the patch embeddings
                patches = model.backbone.embeddings.patch_embeddings(inputs.pixel_values)
                # Apply the backbone's encoder
                encoder_outputs = model.backbone.encoder(patches)
                # Apply the final layernorm to get the final hidden states
                hidden_states = model.backbone.layernorm(encoder_outputs[0])
                
                # Average pooling over the patch dimension to get a global embedding
                batch_embeddings = hidden_states.mean(dim=1)
                all_embeddings.append(batch_embeddings.cpu())
        
        # Combine all embeddings from sub-batches
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        embeddings_np = embeddings_tensor.numpy()
        
        # Save each embedding and metadata to temporary files
        for i, idx in enumerate(valid_indices):
            if i < len(embeddings_np):  # Ensure we don't go out of bounds
                emb_path = os.path.join(temp_embeddings_dir, f"emb_{total_valid_samples}.npy")
                meta_path = os.path.join(temp_metadata_dir, f"meta_{total_valid_samples}.json")
                
                # Save embedding
                np.save(emb_path, embeddings_np[i])
                
                # Save metadata
                global_idx = batch_idx * batch_size + idx
                metadata = {
                    "batch_id": batch_idx,
                    "local_idx": idx,
                    "global_idx": global_idx,
                    "label": batch_labels[idx],
                    "class_name": batch_class_names[idx],
                    "model": f"depth-anything-{model_variant}",
                    "embedding_dim": embeddings_np[i].shape[0],  # Add embedding dimension to metadata
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                total_valid_samples += 1
            
        processed_samples += len(batch_images)
    
    logger.info(f"Total valid samples processed: {total_valid_samples}")
    
    # Create iterators from the saved files
    def embedding_iter():
        for i in range(total_valid_samples):
            emb_path = os.path.join(temp_embeddings_dir, f"emb_{i}.npy")
            yield np.load(emb_path)
            
    def metadata_iter():
        for i in range(total_valid_samples):
            meta_path = os.path.join(temp_metadata_dir, f"meta_{i}.json")
            with open(meta_path, 'r') as f:
                yield json.load(f)
    
    logger.info(f"Creating storage at {storage_dir}")
    
    # Initialize and create storage
    storage = TensorStorage.create_storage(
        storage_dir=storage_dir,
        data_iterator=embedding_iter(),
        metadata_iterator=metadata_iter(),
        description=f"Embeddings from depth-anything-{model_variant}",
    )
    
    # Clean up temporary files
    shutil.rmtree(temp_embeddings_dir)
    shutil.rmtree(temp_metadata_dir)

    logger.info("Storage creation completed")
    return storage

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from Depth Anything model')
    parser.add_argument('--model-variant', type=str, default='small', 
                        choices=['small', 'base', 'large'], help='Model variant to use')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--storage-dir', type=str, default='storages/depth_anything_embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--debug-samples', type=int, default=1000, 
                       help='Number of samples to process in debug mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Create specific storage directory
    model_storage_dir = os.path.join(args.storage_dir, f"depth_anything_{args.model_variant}")
    if args.debug:
        model_storage_dir += "_debug"
    os.makedirs(model_storage_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Extract embeddings
    storage = extract_depth_anything_embeddings(
        model_variant=args.model_variant,
        dataset=dataset,
        storage_dir=model_storage_dir,
        batch_size=args.batch_size,
        device=args.device,
        debug=args.debug,
        debug_samples=args.debug_samples
    )
    
    # Print storage info
    logger.info(f"Storage creation successful for depth-anything-{args.model_variant}!")
    logger.info(f"Storage info:\n{storage}")

if __name__ == "__main__":
    main()