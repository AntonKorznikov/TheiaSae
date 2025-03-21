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
cfg['device'] = 'cuda:0'
cfg = post_init_cfg(cfg)

sae = BatchTopKSAE(cfg)
model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True).to(cfg['device'])
activations_store = ImageActivationsStore(model, cfg)

train_sae(sae, activations_store, cfg)
