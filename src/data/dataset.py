import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from configs import CONFIG, CRITICAL_CLASSES

class BoneMarrowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples, self.all_labels = [], []
        base = self._find_data_dir(root_dir)
        
        self.classes = sorted([d for d in os.listdir(base) 
                              if os.path.isdir(os.path.join(base, d)) and not d.startswith('.')])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp')
        for cls in self.classes:
            cls_path = os.path.join(base, cls)
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(cls_path, '**', ext), recursive=True))
                files.extend(glob.glob(os.path.join(cls_path, ext)))
            files = list(set(files))
            for f in files:
                self.samples.append((f, self.class_to_idx[cls]))
                self.all_labels.append(self.class_to_idx[cls])

    def _find_data_dir(self, root_dir):
        candidates = [root_dir, os.path.join(root_dir, "bone_marrow_cell_dataset")]
        for candidate in candidates:
            if os.path.exists(candidate):
                subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d)) and not d.startswith('.')]
                if len(subdirs) > 5: return candidate
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
                if len(subdirs) > 5: return item_path
        return root_dir

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except:
            img = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label


class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices, transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except:
            img = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label
