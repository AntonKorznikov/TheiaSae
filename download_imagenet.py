from datasets import load_dataset

ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
ds.save_to_disk("data/imagenet_1k_resized_256")