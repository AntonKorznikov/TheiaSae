import sae
from training import train_sae
from sae import BatchTopKSAE
from activation_store import ImageActivationsStore
from config import get_default_cfg, post_init_cfg
from transformers import AutoModel
import wandb


# wandb.login(
#     relogin=True,
#     key='........',
#     )



cfg = get_default_cfg()
cfg["model_name"] = "theaiinstitute/theia-tiny-patch16-224-cdiv"
cfg["dataset_path"] = "data/imagenet_1k_resized_256"
# cfg["dataset_path"] = "evanarlian/imagenet_1k_resized_256"
cfg["sae_type"] = 'batch-topk'
cfg["dict_size"] = 2**15
cfg["top_k"] = 30
cfg['wandb_project'] = 'theia-sae2.1'
cfg['act_size'] = 768
cfg['device'] = 'cuda:4'
cfg["num_tokens"] = 1e8
cfg["model_batch_size"] = 64
cfg["num_batches_in_buffer"] = 500

cfg = post_init_cfg(cfg)
sae = BatchTopKSAE(cfg)

model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True).to(cfg['device'])
activations_store = ImageActivationsStore(model, cfg)

train_sae(sae, activations_store, cfg)
