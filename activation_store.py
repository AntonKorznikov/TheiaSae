import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import torchvision.transforms as T

class ImageActivationsStore:
    def __init__(self, model, cfg: dict):
        self.model = model
        # Load the ImageNet dataset in streaming mode
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
        self.image_column = self._get_image_column()
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.config = cfg
        self.image_size = cfg.get("image_size", 224)
        
        # Define a transformation to ensure images are resized to the correct size
        # and converted to a uint8 tensor with shape (H, W, C) as expected by the model.
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # converts to (C, H, W) float tensor in [0, 1]
            T.Lambda(lambda x: (x * 255).to(torch.uint8)),  # scale to [0, 255] and convert to uint8
            T.Lambda(lambda x: x.permute(1, 2, 0))  # convert to (H, W, C)
        ])

    def _get_image_column(self):
        # Try to infer the correct column name for the image data.
        sample = next(self.dataset)
        if "image" in sample:
            return "image"
        elif "img" in sample:
            return "img"
        else:
            raise ValueError("Dataset must have an 'image' or 'img' column.")

    def get_batch_images(self):
        all_images = []
        # Accumulate enough images for one model batch
        while len(all_images) < self.model_batch_size:
            sample = next(self.dataset)
            image = sample[self.image_column]
            # Apply the transformation to get a tensor of shape (H, W, C)
            image_tensor = self.transform(image)
            all_images.append(image_tensor)
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
        # Fill the buffer with a number of batches as defined in the config.
        for _ in range(self.num_batches_in_buffer):
            batch_images = self.get_batch_images()
            # Assume activations are of shape (B, num_tokens, act_size)
            # Flatten the first two dimensions so that each row corresponds to one token activation.
            activations = self.get_activations(batch_images).reshape(-1, self.config["act_size"])
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        # Create a PyTorch DataLoader over the activation buffer.
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.config["batch_size"],
            shuffle=True
        )

    def next_batch(self):
        # Return the next mini-batch of activations.
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
