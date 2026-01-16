"""
K-Nearest Neighbors (KNN) classifier for trainedml.

This module implements a supervised classification model using the KNN algorithm from scikit-learn.
It exposes a consistent API (fit, predict, evaluate) as required by the trainedml framework.

Examples
--------
>>> model = KNNModel(n_neighbors=3)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
>>> acc = model.evaluate(X_test, y_test)
"""

from .base import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(BaseModel):
    r"""
    K-Nearest Neighbors (KNN) classification model.

    This class wraps scikit-learn's KNeighborsClassifier and exposes a unified API.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    **kwargs :
        Additional keyword arguments passed to KNeighborsClassifier.

    Attributes
    ----------
    model : KNeighborsClassifier
        The underlying scikit-learn estimator.

    Examples
    --------
    >>> model = KNNModel(n_neighbors=3)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> acc = model.evaluate(X, y)
    """
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        """
        Fit the KNN model on training data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the class for new data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        array-like
            Predicted class labels.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Return the accuracy of the model on test data.

        Parameters
        ----------
        X : array-like
            Test data.
        y : array-like
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return self.model.score(X, y)
