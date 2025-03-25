import torch 

def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(4e8),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "dict_size": 2**15,  # 32768
        "device": "cuda",
        "num_batches_in_buffer": 1000,
        "wandb_project": 'tensor-sae',
        "input_unit_norm": False,
        "perf_log_freq": 1000,
        "sae_type": 'batch-topk',
        "n_batches_to_dead": 20,
        "checkpoint_every": 10000,  # Save checkpoint every 10000 batches
        
        # TensorStorage specific configs
        "use_tensor_storage": True,
        "tensor_storage_path": "storages/depth_anything_embeddings/depth_anything_small",
        "act_size": 384,  # Based on the tensor shape in the provided example
        "shuffle": False,  # Disabled shuffle as requested
        
        # (Batch)TopKSAE specific
        "top_k": 30,
        "top_k_aux": 512,
        "aux_penalty": 0,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    # Create a name that reflects using tensor storage
    storage_name = cfg['tensor_storage_path'].split('/')[-1]
    cfg["name"] = f"{storage_name}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg