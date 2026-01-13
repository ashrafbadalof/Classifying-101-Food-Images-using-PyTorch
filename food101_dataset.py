from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class Food101Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"

        self.samples = []
        for class_dir in self.image_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_dir.name))

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted({c for _, c in self.samples}))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[class_name]
        return image, label
    