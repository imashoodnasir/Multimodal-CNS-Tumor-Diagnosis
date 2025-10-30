
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

def save_confusion(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_roc(y_true, y_prob, path):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    fig, ax = plt.subplots(figsize=(6,6))
    if y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    else:
        for c in range(y_prob.shape[1]):
            fpr, tpr, _ = roc_curve((y_true==c).astype(int), y_prob[:,c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {c} AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'k--',linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
