import heapq
import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
import os
from tqdm import tqdm
import argparse
import random
from transformers import AutoModel
from config import get_default_cfg
from sae import BatchTopKSAE
from activation_store import ImageActivationsStore

def main():
    # Configuration
    cfg = get_default_cfg()
    cfg['shuffle'] = False
    cfg['device'] = 'cuda:1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1_100_000)
    parser.add_argument("--num_neurons", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    # Load model and SAE
    vit_model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True).to(cfg['device'])

    sae = BatchTopKSAE(cfg).eval().to(cfg["device"])
    sae.load_state_dict(torch.load("checkpoints/theaiinstitute/theia-base-patch16-224-cdiv_32768_batch-topk_50_0.0002_97655/sae.pt"))

    # Load dataset
    dataset = load_from_disk(cfg["dataset_path"])
    np.random.seed(42)
    image_column = "image" if "image" in dataset.column_names else "img"

    # Select random neurons upfront
    num_neurons = cfg["dict_size"]
    selected_neurons = random.sample(range(num_neurons), args.num_neurons)
    
    # Initialize data structures only for selected neurons
    activation_records = {neuron: [] for neuron in selected_neurons}
    
    # Process images
    activation_store = ImageActivationsStore(vit_model, cfg)
    num_batches = args.num_images // cfg["model_batch_size"]
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing images"):
            images = activation_store.get_batch_images()
            
            # Get SAE activations
            vit_activations = activation_store.get_activations(images)
            sae_activations = sae.encode(vit_activations)

            # Track max activations per image
            batch_size = images.shape[0]
            for img_idx in range(batch_size):
                global_idx = batch_idx * cfg["model_batch_size"] + img_idx
                token_activations = sae_activations[img_idx]
                max_activations = token_activations.max(dim=0).values
                
                # Only process selected neurons
                for neuron_idx in selected_neurons:
                    activation = max_activations[neuron_idx].item()
                    heap = activation_records[neuron_idx]
                    
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, (activation, global_idx))
                    else:
                        if activation > heap[0][0]:
                            heapq.heappop(heap)
                            heapq.heappush(heap, (activation, global_idx))


    # Create output directories and save top images
    base_dir = "neuron_activations"
    os.makedirs(base_dir, exist_ok=True)
    
    for neuron_idx in tqdm(selected_neurons, desc="Saving images"):
        # Get sorted activations from heap
        heap = activation_records[neuron_idx]
        neuron_activations = sorted(heap, key=lambda x: (-x[0], x[1]))
        
        # Create neuron directory
        neuron_dir = os.path.join(base_dir, f"neuron_{neuron_idx}")
        os.makedirs(neuron_dir, exist_ok=True)
        
        # Save top images
        for i, (activation, idx) in enumerate(neuron_activations):
            idx = int(idx)
            image = dataset[idx][image_column]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image) 
            image.save(os.path.join(neuron_dir, f"top_{i+1}_act_{activation:.2f}.png"))

if __name__ == "__main__":
    main()