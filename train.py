import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import kagglehub

from configs import CONFIG
from src.data import BoneMarrowDataset, SubsetWithTransform, get_train_transforms, get_val_transforms, make_balanced_sampler, create_smart_split
from src.model import DinoV3Learner
from src.losses import HierarchicalFocalLossWithCriticalPenalty
from src.utils import MetricsLogger, compute_val_metrics, generate_report


def train_engine():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_logger = MetricsLogger(save_path=f"training_metrics_{timestamp}.json")
    
    path = kagglehub.dataset_download("andrewmvd/bone-marrow-cell-classification") 
    # Original source: Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021). 
    # An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies [Data set].
    # The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.AXH3-T579
    
    print(f"Path to dataset files: {path}")
    
    train_transforms = get_train_transforms(CONFIG['image_size'])
    val_transforms = get_val_transforms(CONFIG['image_size'])
    
    full_ds = BoneMarrowDataset(path, transform=None)
    train_idx, val_idx = create_smart_split(full_ds)
    
    metrics_logger.metrics["dataset"] = {
        "total_samples": len(full_ds),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "num_classes": len(full_ds.classes),
        "class_names": full_ds.classes,
    }
    metrics_logger.save()
    
    train_ds = SubsetWithTransform(full_ds, train_idx, transform=train_transforms)
    val_ds = SubsetWithTransform(full_ds, val_idx, transform=val_transforms)
    
    sampler = make_balanced_sampler(full_ds, train_idx, multiplier=2)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=CONFIG['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(full_ds.all_labels),
                                         y=full_ds.all_labels)
    class_weights = torch.tensor(np.sqrt(class_weights), dtype=torch.float).to(CONFIG['device'])

    model = DinoV3Learner(len(full_ds.classes)).to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-7
    )
    
    criterion = HierarchicalFocalLossWithCriticalPenalty(
        class_names=full_ds.classes,
        alpha=class_weights,
        gamma=CONFIG['gamma'],
        smoothing=CONFIG['smoothing'],
        within_lineage_weight=CONFIG['within_lineage_weight'],
        sequential_bonus=CONFIG['sequential_bonus'],
        critical_penalty=CONFIG['critical_penalty'],
    ).to(CONFIG['device'])

    best_f1, patience_counter = 0, 0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for i, (imgs, lbls) in enumerate(pbar):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            
            logits = model(imgs)
            loss = criterion(logits, lbls) / CONFIG['accumulation_steps']
            loss.backward()
            
            epoch_loss += loss.item() * CONFIG['accumulation_steps']
            num_batches += 1
            
            if (i + 1) % CONFIG['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item() * CONFIG['accumulation_steps']:.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        scheduler.step()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(CONFIG['device'])
                preds = torch.argmax(model(imgs), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.numpy())

        val_metrics = compute_val_metrics(all_labels, all_preds, full_ds.classes, criterion)
        current_lr = scheduler.get_last_lr()[0]
        
        is_best = val_metrics["weighted_f1"] > best_f1
        
        metrics_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_metrics=val_metrics,
            lr=current_lr,
            is_best=is_best
        )
        
        metrics_logger.log_class_report(epoch + 1, full_ds.classes, all_labels, all_preds)
        metrics_logger.log_confusion_matrix(epoch + 1, full_ds.classes, all_labels, all_preds)
        
        print(f"\n Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | "
              f"Weighted F1: {val_metrics['weighted_f1']:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f} | "
              f"Critical Errors: {val_metrics['critical_errors']} | "
              f"LR: {current_lr:.2e}")
        generate_report(all_labels, all_preds, full_ds.classes)

        if is_best:
            best_f1 = val_metrics["weighted_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n Early stopping at epoch {epoch+1}")
                metrics_logger.finalize(early_stopped=True, final_epoch=epoch+1)
                break
    else:
        metrics_logger.finalize(early_stopped=False, final_epoch=CONFIG['epochs'])

    print(f"\n Done! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train_engine()