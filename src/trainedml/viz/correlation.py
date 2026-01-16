"""
Correlation analysis utilities for trainedml.

This module provides functions and classes for computing and visualizing correlation matrices
between variables, supporting different correlation methods and visual outputs.

Examples
--------
>>> from trainedml.viz.correlation import correlation_matrix
>>> corr = correlation_matrix(df)
>>> print(corr)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .vizs import Vizs

def correlation_matrix(data, features='all', method='pearson'):
    """
    Compute the correlation matrix for selected features.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    features : 'all' or list, default='all'
        Features to include.
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall').

    Returns
    -------
    pandas.DataFrame
        Correlation matrix.

    Examples
    --------
    >>> corr = correlation_matrix(df, features=['A', 'B'], method='spearman')
    >>> print(corr)
    """
    cols = data.columns.tolist() if features == 'all' else features
    return data[cols].corr(method=method)

class CorrelationViz(Vizs):
    """
    Classe pour générer une heatmap de corrélation.
    """
    def __init__(self, data, features='all', method='pearson', mask=True):
        super().__init__(data)
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self):
        if self._features == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._features
        corr = self._data[cols].corr(method=self._method)
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Matrice de corrélation')
        self._figure = fig
