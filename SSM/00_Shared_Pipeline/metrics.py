import numpy as np
from sklearn.metrics import log_loss, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

def plasticc_log_loss(y_true, y_preds_proba, class_labels):
    """
    Calculates the official Kaggle PLAsTiCC weighted log-loss.
    Rare/anomaly classes (like 64/Kilonova and 99/Other) are penalized 
    twice as heavily (weight = 2) as common classes (weight = 1).
    
    Parameters:
    - y_true: array-like of true class labels (integer IDs)
    - y_preds_proba: 2D array of predicted probabilities (shape: [n_samples, n_classes])
    - class_labels: list of all possible class integer IDs matching the columns of y_preds_proba
    """
    # PLAsTiCC specific weights: 2 for classes 64 and 99, 1 for all others
    weights = {cls: 2 if cls in [64, 99] else 1 for cls in class_labels}
    
    # Calculate the log-loss for each class independently
    class_losses = []
    class_weights = []
    
    for i, cls_label in enumerate(class_labels):
        # Create a binary mask for the current class
        y_true_binary = (np.array(y_true) == cls_label).astype(int)
        preds_for_class = y_preds_proba[:, i]
        
        # Calculate standard log-loss for this binary outcome
        # eps is added automatically by sklearn to prevent log(0)
        loss = log_loss(y_true_binary, preds_for_class, labels=[0, 1])
        
        class_losses.append(loss)
        class_weights.append(weights[cls_label])
        
    # Compute the weighted average
    weighted_log_loss = np.average(class_losses, weights=class_weights)
    return weighted_log_loss

def macro_f1(y_true, y_pred_classes):
    """
    Calculates the Macro F1-Score to evaluate performance across highly imbalanced classes.
    
    Parameters:
    - y_true: array-like of true class labels
    - y_pred_classes: array-like of predicted class labels (the argmax of probabilities)
    """
    return f1_score(y_true, y_pred_classes, average='macro', zero_division=0)

def macro_pr_auc(y_true, y_preds_proba, class_labels):
    """
    Calculates the Macro-Averaged Precision-Recall Area Under the Curve (PR-AUC).
    Highly robust against the massive "True Negative" skew of the SNIa class.
    
    Parameters:
    - y_true: array-like of true class labels
    - y_preds_proba: 2D array of predicted probabilities
    - class_labels: list of all possible class integer IDs
    """
    # Convert true labels to one-hot encoded matrix for sklearn's PR function
    y_true_one_hot = label_binarize(y_true, classes=class_labels)
    
    # Calculate the unweighted average of the PR-AUC across all classes
    return average_precision_score(y_true_one_hot, y_preds_proba, average='macro')

def multiclass_brier_score(y_true, y_preds_proba, class_labels):
    """
    Calculates the Multiclass Brier Score to measure probabilistic calibration.
    A lower score (closer to 0) means the model's confidence perfectly matches reality.
    
    Parameters:
    - y_true: array-like of true class labels
    - y_preds_proba: 2D array of predicted probabilities
    - class_labels: list of all possible class integer IDs
    """
    # Convert true labels to one-hot format
    y_true_one_hot = label_binarize(y_true, classes=class_labels)
    
    # Brier score is the mean squared error between predicted probabilities and one-hot true targets
    return np.mean(np.sum((y_preds_proba - y_true_one_hot)**2, axis=1))