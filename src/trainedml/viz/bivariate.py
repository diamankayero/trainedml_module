"""
Bivariate analysis visualization for trainedml.

This module provides the BivariateViz class, which generates scatter plots and other
bivariate visualizations between two variables using matplotlib.

Examples
--------
>>> from trainedml.viz.bivariate import BivariateViz
>>> viz = BivariateViz(df, x='A', y='B')
>>> viz.vizs()
>>> viz.figure.show()
"""

import matplotlib.pyplot as plt

class BivariateViz:
    r"""
    Bivariate analysis visualization (scatter plot, etc.).

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    x : str
        First variable (x-axis).
    y : str
        Second variable (y-axis).

    Attributes
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        X-axis variable.
    y : str
        Y-axis variable.
    figure : matplotlib.figure.Figure
        The generated figure (after calling vizs).

    Examples
    --------
    >>> viz = BivariateViz(df, x='A', y='B')
    >>> viz.vizs()
    >>> viz.figure.show()
    """
    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y
        self.figure = None

    def vizs(self):
        """
        Generate the bivariate scatter plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated scatter plot figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.data[self.x], self.data[self.y], alpha=0.7)
        ax.set_title(f'Scatter Plot: {self.x} vs {self.y}')
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        self.figure = fig
        return fig
