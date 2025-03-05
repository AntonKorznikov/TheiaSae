import torch 

def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "model_name": "theaiinstitute/theia-base-patch16-224-cdiv",
        "act_size": 768,
        "dict_size": 12288,
        "device": "cuda:4",
        "model_batch_size": 512,
        "num_batches_in_buffer": 200,
        "dataset_path": "evanarlian/imagenet_1k_resized_256",
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": False,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "n_batches_to_dead": 20,

        # (Batch)TopKSAE specific
        "top_k": 30,
        "top_k_aux": 512,
        "aux_penalty": 0.0,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg