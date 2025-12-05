import json
from datetime import datetime
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from configs import CONFIG, CRITICAL_CLASSES, SAFE_CLASSES

class MetricsLogger:
    def __init__(self, save_path="training_metrics.json"):
        self.save_path = save_path
        self.metrics = {
            "config": CONFIG.copy(),
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "best_epoch": None,
            "best_weighted_f1": 0,
            "training_complete": False,
        }
        self.metrics["config"]["device"] = str(CONFIG["device"])
    
    def log_epoch(self, epoch, train_loss, val_metrics, lr, is_best=False):
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "learning_rate": lr,
            "is_best": is_best,
            **val_metrics
        }
        self.metrics["epochs"].append(epoch_data)
        
        if is_best:
            self.metrics["best_epoch"] = epoch
            self.metrics["best_weighted_f1"] = val_metrics["weighted_f1"]
        
        self.save()
    
    def log_class_report(self, epoch, class_names, y_true, y_pred):
        report = classification_report(
            y_true, y_pred, target_names=class_names,
            labels=range(len(class_names)), output_dict=True, zero_division=0
        )
        for ep in self.metrics["epochs"]:
            if ep["epoch"] == epoch:
                ep["per_class_metrics"] = {
                    cls: {
                        "precision": report[cls]["precision"],
                        "recall": report[cls]["recall"],
                        "f1": report[cls]["f1-score"],
                        "support": report[cls]["support"]
                    } for cls in class_names if cls in report
                }
                break
        self.save()
    
    def log_confusion_matrix(self, epoch, class_names, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        for ep in self.metrics["epochs"]:
            if ep["epoch"] == epoch:
                ep["confusion_matrix"] = cm.tolist()
                ep["class_names"] = class_names
                break
        self.save()
    
    def finalize(self, early_stopped=False, final_epoch=None):
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["training_complete"] = True
        self.metrics["early_stopped"] = early_stopped
        self.metrics["final_epoch"] = final_epoch
        self.save()
        print(f"\n Metrics saved to: {self.save_path}")
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def compute_val_metrics(y_true, y_pred, class_names, criterion):
    metrics = {
        "weighted_f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "critical_errors": criterion.get_critical_error_count(y_true, y_pred, class_names),
        "total_samples": len(y_true),
    }
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    errors = [(class_names[i], class_names[j], int(cm[i,j]))
              for i in range(len(class_names)) for j in range(len(class_names))
              if i != j and cm[i,j] > 0]
    errors.sort(key=lambda x: -x[2])
    metrics["top_confusions"] = [{"true": t, "pred": p, "count": c} for t, p, c in errors[:10]]
    
    return metrics


def generate_report(y_true, y_pred, class_names):
    print("\n" + "=" * 50)
    print(classification_report(
        y_true, y_pred, target_names=class_names,
        labels=range(len(class_names)), digits=4, zero_division=0
    ))
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    errors = [(class_names[i], class_names[j], cm[i,j])
              for i in range(len(class_names)) for j in range(len(class_names))
              if i != j and cm[i,j] > 0]
    errors.sort(key=lambda x: -x[2])
    
    if errors:
        print("--- Top Confusions ---")
        for true, pred, cnt in errors[:5]:
            print(f"  CRITICAL {true} → {pred}: {cnt}")
    
    critical_idx = set(i for i, n in enumerate(class_names) if n in CRITICAL_CLASSES)
    safe_idx = set(i for i, n in enumerate(class_names) if n in SAFE_CLASSES)
    
    critical_errors = []
    for i in critical_idx:
        for j in safe_idx:
            if cm[i, j] > 0:
                critical_errors.append((class_names[i], class_names[j], cm[i, j]))
    
    if critical_errors:
        critical_errors.sort(key=lambda x: -x[2])
        print("\n CRITICAL ERRORS (Cancer → Normal):")
        for true, pred, cnt in critical_errors:
            print(f" CRITICAL! {true} → {pred}: {cnt}")
        print(f"  Total critical errors: {sum(x[2] for x in critical_errors)}")
