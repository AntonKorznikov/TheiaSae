import sae
from training import train_sae
from sae import BatchTopKSAE
from activation_store import ImageActivationsStore
from config import get_default_cfg, post_init_cfg
from transformers import AutoModel


cfg = get_default_cfg()
cfg["model_name"] = "theaiinstitute/theia-base-patch16-224-cdiv"
cfg["dataset_path"] = "evanarlian/imagenet_1k_resized_256"
cfg["sae_type"] = 'batch-topk'
cfg["dict_size"] = 2**16
cfg["top_k"] = 30
cfg['wandb_project'] = 'theia-sae'
cfg['act_size'] = 2304
cfg['device'] = 'cuda:0'
cfg["num_tokens"] = 5e8
cfg["model_batch_size"] = 32

cfg = post_init_cfg(cfg)
sae = BatchTopKSAE(cfg)

model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True)
activations_store = ImageActivationsStore(model, cfg)

train_sae(sae, activations_store, cfg)
