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
    Visualisation de courbe entre deux colonnes.

    Paramètres
    ----------
    data : pandas.DataFrame
        Le jeu de données.
    x_column : str
        Colonne pour l'axe des x.
    y_column : str
        Colonne pour l'axe des y.
    save_path : str ou None
        Chemin de sauvegarde optionnel.
    """
    def __init__(self, data: pd.DataFrame, x_column: str, y_column: str, save_path: Optional[str] = None):
        super().__init__(data, save_path)
        self._x_column = x_column
        self._y_column = y_column

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
