"""
Line plot visualization for trainedml.

This module provides the LineViz class, which generates line plots between two columns
using matplotlib, supporting custom styling and axis labeling.

Examples
--------
>>> from trainedml.viz.line import LineViz
>>> viz = LineViz(df, x_column='A', y_column='B')
>>> viz.vizs()
>>> viz.figure.show()
"""

import matplotlib.pyplot as plt
from typing import Optional
from .vizs import Vizs


class LineViz(Vizs):
    r"""
    Line plot visualization between two columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    x_column : str
        Column for the x-axis.
    y_column : str
        Column for the y-axis.

    Attributes
    ----------
    data : pandas.DataFrame
        The data.
    x_column : str
        X-axis column.
    y_column : str
        Y-axis column.
    figure : matplotlib.figure.Figure
        The generated figure (after calling vizs).

    Examples
    --------
    >>> viz = LineViz(df, x_column='A', y_column='B')
    >>> viz.vizs()
    >>> viz.figure.show()
    """
    def __init__(self, data, x_column, y_column, save_path: Optional[str] = None):
        super().__init__(data, save_path=save_path)
        self.x_column = x_column
        self.y_column = y_column

    def vizs(self):
        """
        Generate the line plot figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated line plot figure.
        """
        plt.figure(figsize=(8, 6))
        self._figure = plt.plot(self._data[self.x_column], self._data[self.y_column], marker='o')
        plt.title(f"Courbe {self.y_column} en fonction de {self.x_column}")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.tight_layout()
        self._auto_save()
