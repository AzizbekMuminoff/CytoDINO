import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import LINEAGE_GROUPS, SEQUENTIAL_LINEAGES, CRITICAL_CLASSES, SAFE_CLASSES

class HierarchicalFocalLossWithCriticalPenalty(nn.Module):
    def __init__(self, class_names, alpha=None, gamma=2.0, smoothing=0.1,
                 within_lineage_weight=0.7, sequential_bonus=0.5, critical_penalty=3.0):
        super().__init__()
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.gamma = gamma
        self.alpha = alpha
        self.critical_penalty = critical_penalty
        
        smooth_matrix = self._build_smoothing_matrix(class_names, smoothing, 
                                                      within_lineage_weight, sequential_bonus)
        self.register_buffer('smooth_matrix', smooth_matrix)
        critical_mask = self._build_critical_mask(class_names)
        self.register_buffer('critical_mask', critical_mask)
        self._print_setup_info(class_names)
    
    def _build_smoothing_matrix(self, class_names, smoothing, within_w, seq_bonus):
        C = self.num_classes
        matrix = torch.zeros(C, C)
        class_to_lineage = {}
        for lineage, classes in LINEAGE_GROUPS.items():
            for cls in classes:
                class_to_lineage[cls] = lineage
        
        for i, name_i in enumerate(class_names):
            lineage_i = class_to_lineage.get(name_i, "unknown")
            same_lineage, adjacent_seq, other_classes = [], [], []
            
            for j, name_j in enumerate(class_names):
                if i == j: continue
                lineage_j = class_to_lineage.get(name_j, "unknown")
                if lineage_i == lineage_j and lineage_i != "unknown":
                    same_lineage.append(j)
                    if lineage_i in SEQUENTIAL_LINEAGES:
                        seq = SEQUENTIAL_LINEAGES[lineage_i]
                        if name_i in seq and name_j in seq:
                            if abs(seq.index(name_i) - seq.index(name_j)) == 1:
                                adjacent_seq.append(j)
                else:
                    other_classes.append(j)
            
            within_budget = smoothing * within_w
            cross_budget = smoothing * (1 - within_w)
            
            if same_lineage:
                if adjacent_seq:
                    adj_budget = within_budget * seq_bonus
                    other_within = within_budget * (1 - seq_bonus)
                    for j in adjacent_seq:
                        matrix[i, j] = adj_budget / len(adjacent_seq)
                    non_adj = [j for j in same_lineage if j not in adjacent_seq]
                    if non_adj:
                        for j in non_adj:
                            matrix[i, j] = other_within / len(non_adj)
                else:
                    for j in same_lineage:
                        matrix[i, j] = within_budget / len(same_lineage)
            else:
                cross_budget += within_budget
            
            if other_classes:
                for j in other_classes:
                    matrix[i, j] = cross_budget / len(other_classes)
            matrix[i, i] = 1.0 - smoothing
        return matrix
    
    def _build_critical_mask(self, class_names):
        C = self.num_classes
        mask = torch.ones(C, C)
        critical_idx = [i for i, n in enumerate(class_names) if n in CRITICAL_CLASSES]
        safe_idx = [i for i, n in enumerate(class_names) if n in SAFE_CLASSES]
        for i in critical_idx:
            for j in safe_idx:
                mask[i, j] = self.critical_penalty
        for i in safe_idx:
            for j in critical_idx:
                mask[i, j] = 1.5
        return mask
    
    def _print_setup_info(self, class_names):
        critical_idx = [i for i, n in enumerate(class_names) if n in CRITICAL_CLASSES]
        safe_idx = [i for i, n in enumerate(class_names) if n in SAFE_CLASSES]
        print("\n" + "="*60)
        print("="*60)
        print(f"Critical classes: {[class_names[i] for i in critical_idx]}")
        print(f"Safe classes: {[class_names[i] for i in safe_idx]}")
        print(f"Critical→Safe penalty: {self.critical_penalty}x")
        print("="*60 + "\n")
    
    def forward(self, inputs, targets):
        B = inputs.shape[0]
        soft_labels = self.smooth_matrix[targets]
        probs = F.softmax(inputs, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        ce = -(soft_labels * log_probs).sum(dim=-1)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce = alpha_t * ce
        preds = torch.argmax(probs, dim=1)
        critical_weights = self.critical_mask[targets, preds]
        return (focal_weight * ce * critical_weights).mean()
    
    def get_critical_error_count(self, targets, preds, class_names):
        critical_idx = set(i for i, n in enumerate(class_names) if n in CRITICAL_CLASSES)
        safe_idx = set(i for i, n in enumerate(class_names) if n in SAFE_CLASSES)
        return sum(1 for t, p in zip(targets, preds) if t in critical_idx and p in safe_idx)
