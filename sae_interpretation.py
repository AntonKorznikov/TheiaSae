import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
import os
from tqdm import tqdm
import argparse
import random
from transformers import AutoModel
from config import cfg
from sae import BatchTopKSAE
from activation_store import ImageActivationsStore

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100_000)
    parser.add_argument("--num_neurons", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    # Load model and SAE
    vit_model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True).to(cfg['device'])

    sae = BatchTopKSAE(cfg).eval().to(cfg["device"])
    sae.load_state_dict(torch.load("checkpoints/sae.pt"))

    # Load dataset and create permutation
    dataset = load_from_disk(cfg["dataset_path"])
    np.random.seed(42)
    permuted_indices = np.random.permutation(len(dataset))
    image_column = "image" if "image" in dataset.column_names else "img"

    # Initialize data structures
    num_neurons = cfg["dict_size"]
    activation_records = {i: [] for i in range(num_neurons)}
    
    # Process images
    activation_store = ImageActivationsStore(vit_model, cfg)
    num_batches = args.num_images // cfg["model_batch_size"]
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Processing images"):
            images = activation_store.get_batch_images()
            
            # Get SAE activations
            vit_activations = activation_store.get_activations(images)
            sae_activations = sae.encode(vit_activations)
            
            # Track max activations per image
            batch_size = images.shape[0]
            num_tokens = vit_activations.shape[1]
            for img_idx in range(batch_size):
                global_idx = _ * cfg["model_batch_size"] + img_idx
                token_activations = sae_activations[img_idx*num_tokens:(img_idx+1)*num_tokens]
                max_activations = token_activations.max(dim=0).values
                
                for neuron_idx in range(num_neurons):
                    activation = max_activations[neuron_idx].item()
                    activation_records[neuron_idx].append((activation, global_idx))

    # Select random neurons
    selected_neurons = random.sample(range(num_neurons), args.num_neurons)

    # Create output directories and save top images
    base_dir = "neuron_activations"
    os.makedirs(base_dir, exist_ok=True)
    
    for neuron_idx in tqdm(selected_neurons, desc="Saving images"):
        # Sort and select top images
        neuron_activations = sorted(activation_records[neuron_idx], 
                                 key=lambda x: x[0], reverse=True)[:args.top_k]
        
        # Create neuron directory
        neuron_dir = os.path.join(base_dir, f"neuron_{neuron_idx}")
        os.makedirs(neuron_dir, exist_ok=True)
        
        # Save top images
        for i, (activation, idx) in enumerate(neuron_activations):
            original_idx = permuted_indices[idx]
            image = dataset[original_idx][image_column]
            
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            image.save(os.path.join(neuron_dir, f"top_{i+1}_act_{activation:.2f}.png"))

if __name__ == "__main__":
    main()