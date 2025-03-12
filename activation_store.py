import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset, load_from_disk
import torchvision.transforms as T
from tqdm import tqdm
import torchvision.utils as vutils
import wandb
import torchvision

class ImageActivationsStore:
    def __init__(self, model, cfg: dict, wandb_run=None):
        self.model = model
        self.wandb_run = wandb_run 
        self.dataset = iter(load_from_disk(cfg["dataset_path"]).shuffle(seed=42))
        self.image_column = self._get_image_column()
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.config = cfg
        self.image_size = cfg.get("image_size", 224)
        self.current_step = 0
        self.seen_images = 0
        self.seen_rgb_images = 0
        self.cfg = cfg

        # Initialize the transformation pipeline
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # Converts PIL.Image to (C, H, W) tensor in [0, 1]
            T.Lambda(lambda x: (x * 255).to(torch.uint8)),  # Scale to [0, 255]
            T.Lambda(lambda x: x.permute(1, 2, 0)),  # Change to (H, W, C)
        ])

        # Initialize activation buffer, dataloader, and iterator
        self.activation_buffer = self._fill_buffer()
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _get_image_column(self):
        # Try to infer the correct column name for the image data.
        sample = next(self.dataset)
        if "image" in sample:
            return "image"
        elif "img" in sample:
            return "img"
        else:
            raise ValueError("Dataset must have an 'image' or 'img' column.")
        
    def set_current_step(self, step: int):
        self.current_step = step


    def get_batch_images(self):
        all_images = []
        # Accumulate enough images for one model batch
        while len(all_images) < self.model_batch_size:
            try:
                sample = next(self.dataset)
            except:
                self.dataset = iter(load_from_disk(self.cfg["dataset_path"]).shuffle(seed=43))
                sample = next(self.dataset)

            self.seen_images += 1
                
            image = sample[self.image_column]
            # Apply the transformation to get a tensor of shape (H, W, C)
            image_tensor = self.transform(image)
            if image_tensor.shape[-1] != 3:
                continue  # Skip this image
            all_images.append(image_tensor)
            self.seen_rgb_images += 1
        # Stack images into a single tensor and move to the proper device.
        # The resulting shape is (B, H, W, C)
        batch = torch.stack(all_images, dim=0).to(self.device)
        return batch

    def get_activations(self, batch_images: torch.Tensor):
        with torch.no_grad():
            # Extract the intermediate features (which include patch tokens and [CLS] token)
            activations = self.model.forward_feature(batch_images)
        return activations

    def _fill_buffer(self):
        all_activations = []
        for i in tqdm(range(self.num_batches_in_buffer), desc="Processing batches"):
            batch_images = self.get_batch_images()
            
            # New: Log images to WandB
            if self.wandb_run and i % self.config.get("log_images_every_buffer", 5) == 0:
                with torch.no_grad():
                    # Convert from (B, H, W, C) to (B, C, H, W)
                    log_images = batch_images[:self.config['max_buffer_images']].permute(0, 3, 1, 2).cpu()
                    
                    # Create image grid
                    grid = torchvision.utils.make_grid(
                        log_images,
                        nrow=self.config['max_buffer_images'] // 5,
                        normalize=False,
                        scale_each=False
                    )
                    
                    # Log to WandB
                    self.wandb_run.log({
                        "buffer_images": wandb.Image(grid, caption=f"Buffer Batch {i}")
                    }, step = self.current_step)

                    self.wandb_run.log({
                        "number_of_seen_images": self.seen_images
                    }, step = self.current_step)

                    self.wandb_run.log({
                        "number_of_seen_rgb_images": self.seen_rgb_images
                    }, step = self.current_step)
                    

            

            # Existing processing code
            activations = self.get_activations(batch_images).reshape(-1, self.config["act_size"])
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.config["batch_size"],
            shuffle=True,  
        )
    def next_batch(self):
        torch.cuda.empty_cache()
        try:
            batch = next(self.dataloader_iter)
        except:
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
            # Extract the tensor from the tuple (TensorDataset returns batches as tuples)
        return batch[0]
