import random
import numpy as np
import torch

class BalancedOversampler(torch.utils.data.Sampler):
    def __init__(self, weights, n_samples, multiplier):
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.n_samples = n_samples
        self.multiplier = multiplier
        
    def __iter__(self):
        base = list(range(self.n_samples))
        num_extra = int(self.n_samples * (self.multiplier - 1))
        extra = torch.multinomial(self.weights, num_extra, replacement=True).tolist()
        combined = base + extra
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return int(self.n_samples * self.multiplier)


def make_balanced_sampler(full_ds, train_indices, multiplier=4):
    print(f"\n Creating Weighted Sampler...")
    labels = [full_ds.all_labels[i] for i in train_indices]
    class_counts = np.bincount(labels, minlength=len(full_ds.classes))
    weights = 1.0 / np.power(class_counts + 1, 0.7) # change factor if you have a stronger class disbalance
    weights[class_counts == 0] = 0
    sample_weights = [weights[l] for l in labels]
    return BalancedOversampler(sample_weights, len(train_indices), multiplier)


def create_smart_split(full_ds):
    class_indices = {}
    for idx, label in enumerate(full_ds.all_labels):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for cls_idx in sorted(class_indices.keys()):
        indices = class_indices[cls_idx].copy()
        np.random.shuffle(indices)
        n = len(indices)
        if n == 1:
            val_idx.extend(indices)
            train_idx.extend(indices)
        elif n <= 5:
            val_idx.append(indices[0])
            train_idx.extend(indices[1:])
        elif n < 20:
            n_val = max(1, n // 5)
            val_idx.extend(indices[:n_val])
            train_idx.extend(indices[n_val:])
        else:
            n_val = int(n * 0.2) # 20% Validation split
            val_idx.extend(indices[:n_val])
            train_idx.extend(indices[n_val:])

    print(f"\n Total: {len(train_idx)} train / {len(val_idx)} val")
    return train_idx, val_idx