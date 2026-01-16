"""
Evaluation utilities for classification models in trainedml.

This module provides standard metrics for evaluating classification models, such as accuracy,
precision, recall, and F1-score, using scikit-learn metrics.

Mathematical Formulation
------------------------
Let $y_i$ be the true label and $\hat{y}_i$ the predicted label for sample $i$.

- **Accuracy**:

  .. math::
      \mathrm{accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i = \hat{y}_i)

- **Precision** (for class $k$):

  .. math::
      \mathrm{precision}_k = \frac{TP_k}{TP_k + FP_k}

- **Recall** (for class $k$):

  .. math::
      \mathrm{recall}_k = \frac{TP_k}{TP_k + FN_k}

- **F1-score** (for class $k$):

  .. math::
      \mathrm{F1}_k = 2 \cdot \frac{\mathrm{precision}_k \cdot \mathrm{recall}_k}{\mathrm{precision}_k + \mathrm{recall}_k}

where $TP_k$, $FP_k$, $FN_k$ are the true positives, false positives, and false negatives for class $k$.

Examples
--------
>>> from trainedml.evaluation import Evaluator
>>> y_true = [0, 1, 1, 0]
>>> y_pred = [0, 1, 0, 0]
>>> scores = Evaluator.evaluate_all(y_true, y_pred)
>>> print(scores)
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    r"""
    Utility class for evaluating classification models.

    Provides static methods to compute standard classification metrics.
    """
    @staticmethod
    def evaluate_all(y_true, y_pred):
        """
        Compute accuracy, precision, recall, and F1-score for classification.

        Parameters
        ----------
        y_true : array-like
            True class labels.
        y_pred : array-like
            Predicted class labels.

        Returns
        -------
        dict
            Dictionary with keys 'accuracy', 'precision', 'recall', 'f1'.

        Examples
        --------
        >>> Evaluator.evaluate_all([0, 1, 1], [0, 1, 0])
        {'accuracy': 0.666..., 'precision': 0.666..., 'recall': 0.666..., 'f1': 0.666...}
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
