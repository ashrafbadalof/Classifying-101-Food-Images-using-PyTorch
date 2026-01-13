from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import json

def idx_food101(root_dir):
    root_dir = Path(root_dir) / "images"
    samples = []

    for class_dir in sorted(root_dir.iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                samples.append((str(img_path), class_dir.name))
    classes = sorted({label for _, label in samples})
    class_to_idx = {cls:idx for idx, cls in enumerate(classes)}

    return samples, class_to_idx

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

class Food101Dataset(Dataset):
    def __init__(self, samples, class_to_idx, transform = None):

        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label
    