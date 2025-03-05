import sae
from training import train_sae
from sae import BatchTopKSAE
from activation_store import ImageActivationsStore
from config import get_default_cfg, post_init_cfg
from transformers import AutoModel
import wandb
from logs import init_wandb, log_wandb, save_checkpoint

wandb.login(
     relogin=True,
     key='0e3060bf75b4fe4a4f986dc128128b15ff962637',
     )



cfg = get_default_cfg()
cfg["model_name"] = "theaiinstitute/theia-tiny-patch16-224-cdiv"
cfg["dataset_path"] = "data/imagenet_1k_resized_256"
# cfg["dataset_path"] = "evanarlian/imagenet_1k_resized_256"
cfg["sae_type"] = 'batch-topk'
cfg["dict_size"] = 2**15
cfg["top_k"] = 30
cfg['wandb_project'] = 'theia-sae2.1'
cfg['act_size'] = 768
cfg['device'] = 'cuda:0'
cfg["num_tokens"] = 1e8
cfg["model_batch_size"] = 64
cfg["num_batches_in_buffer"] = 300
cfg["log_images_every_buffer"] = 10  # Log every 100th batch during buffer filling
cfg["max_buffer_images"] = 25

cfg = post_init_cfg(cfg)
sae = BatchTopKSAE(cfg)

model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True).to(cfg['device'])


wandb_run = init_wandb(cfg)

activations_store = ImageActivationsStore(model, cfg, wandb_run)
train_sae(sae, activations_store, cfg, wandb_run)
