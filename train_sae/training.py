import torch
import tqdm
from logs import init_wandb, log_wandb, save_checkpoint

def train_sae(sae, activation_store, cfg):
    """
    Train a sparse autoencoder using activations from the provided activation store.
    
    Args:
        sae: The sparse autoencoder model
        activation_store: Store providing batches of activations
        cfg: Configuration dictionary
    """
    # Determine number of batches based on total embeddings or configured tokens
    if hasattr(activation_store, 'total_samples'):
        # Calculate based on actual number of samples in TensorStorage
        num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
        print(f"Training for {num_batches} batches based on {activation_store.total_samples} total samples")
    else:
        # Fall back to the configured number of tokens
        num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
        print(f"Training for {num_batches} batches based on configured token count")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        sae.parameters(), 
        lr=cfg["lr"], 
        betas=(cfg["beta1"], cfg["beta2"])
    )
    
    # Create progress bar
    pbar = tqdm.trange(num_batches)

    # Initialize wandb logging
    wandb_run = init_wandb(cfg)
    
    # Training loop
    for i in pbar:
        # Get next batch of activations
        batch = activation_store.next_batch()
        
        # Forward pass through SAE
        sae_output = sae(batch)
        
        # Log metrics
        log_wandb(sae_output, i, wandb_run)

        # Extract and display loss
        loss = sae_output["loss"]
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "L0": f"{sae_output['l0_norm']:.4f}", 
            "L2": f"{sae_output['l2_loss']:.4f}", 
            "L1": f"{sae_output['l1_loss']:.4f}", 
            "L1_norm": f"{sae_output['l1_norm']:.4f}"
        })
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save checkpoint periodically
        if (i + 1) % cfg.get("checkpoint_every", 10000) == 0:
            print(f"Saving checkpoint at batch {i+1}")
            save_checkpoint(wandb_run, sae, cfg, i)

    # Save final checkpoint
    print("Training complete, saving final checkpoint")
    save_checkpoint(wandb_run, sae, cfg, i)