"""
Boxplot visualization for trainedml.

This module provides the BoxplotViz class, which generates boxplots for one or more columns
using matplotlib, supporting grouping by another variable.

Examples
--------
>>> from trainedml.viz.boxplot import BoxplotViz
>>> viz = BoxplotViz(df, columns=['A', 'B'])
>>> viz.vizs()
>>> viz.figure.show()
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class BoxplotViz(Vizs):
    r"""
    Boxplot visualization for one or more columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    columns : 'all' or list, default='all'
        Columns to plot.
    by : str or None, default=None
        Grouping variable.

    Attributes
    ----------
    data : pandas.DataFrame
        The data.
    columns : list
        Columns used.
    by : str or None
        Grouping variable.
    figure : matplotlib.figure.Figure
        The generated figure (after calling vizs).

    Examples
    --------
    >>> viz = BoxplotViz(df, columns=['A', 'B'], by='Group')
    >>> viz.vizs()
    >>> viz.figure.show()
    """
    def __init__(self, data, columns='all', by=None):
        super().__init__(data)
        self._columns = columns
        self._by = by

    def vizs(self):
        """
        Generate the boxplot figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot figure.
        """
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            if self._by:
                self._data.boxplot(column=col, by=self._by, ax=ax)
                ax.set_title(f"Boxplot de {col} par {self._by}")
            else:
                ax.boxplot(self._data[col].dropna(), vert=False)
                ax.set_title(f"Boxplot de {col}")
        plt.tight_layout()
        self._figure = fig
        return fig
