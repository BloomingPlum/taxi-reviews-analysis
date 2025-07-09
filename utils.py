import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import torch.nn.functional as F

# Metric computation function for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro'),
        'recall_macro': recall_score(labels, preds, average='macro'),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }
    # Only compute ROC AUC if all classes are present in labels
    try:
        if len(np.unique(labels)) == probs.shape[1]:
            metrics['roc_auc_ovr'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        else:
            metrics['roc_auc_ovr'] = None
    except Exception as e:
        metrics['roc_auc_ovr'] = None
    return metrics

# Explanation function using transformers-interpret
def explain_sentence(sentence, explainer):
    attributions = explainer(sentence)  # returns list of (token, score)
    sorted_attributions = sorted(attributions, key=lambda x: x[1], reverse=True)

    print(f"\nExplanation for: {sentence}")
    for token, score in sorted_attributions:
        print(f"{token}: {score:.4f}")