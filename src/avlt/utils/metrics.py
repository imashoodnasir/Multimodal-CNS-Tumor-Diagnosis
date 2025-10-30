
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

class MetricTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.preds, self.targets, self.probs = [], [], []

    def update(self, logits, y):
        import torch
        p = logits.softmax(dim=1).argmax(dim=1)
        self.preds.append(p.cpu().numpy())
        self.targets.append(y.cpu().numpy())
        self.probs.append(logits.softmax(dim=1).cpu().numpy())

    def report(self):
        y = np.concatenate(self.targets)
        p = np.concatenate(self.preds)
        prob = np.concatenate(self.probs)
        acc = accuracy_score(y, p)
        f1 = f1_score(y, p, average='macro')
        res = {"acc": float(acc), "f1": float(f1)}
        if self.num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y, prob[:,1])
                res["auc"] = float(auc)
            except Exception:
                pass
        else:
            try:
                auc = roc_auc_score(y, prob, multi_class='ovr')
                res["auc_macro"] = float(auc)
            except Exception:
                pass
        res["confusion"] = confusion_matrix(y, p).tolist()
        return res
