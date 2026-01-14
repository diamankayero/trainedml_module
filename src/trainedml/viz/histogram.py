"""
Visualisation d'un histogramme pour trainedml.
Permet de tracer un ou plusieurs histogrammes à partir d'un DataFrame pandas.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .vizs import Vizs


class HistogramViz(Vizs):
    """
    Classe pour générer un ou plusieurs histogrammes à partir d'une ou plusieurs colonnes.
    
    Args:
        data: DataFrame pandas
        columns: 'all' ou liste de colonnes à tracer
        legend: Afficher la légende
        bins: Nombre de classes
        save_path: Chemin optionnel pour sauvegarder la figure
    """
    def __init__(self, data, columns='all', legend=False, bins=10, save_path: Optional[str] = None):
        super().__init__(data, save_path=save_path)
        # Vérification des arguments
        if not isinstance(columns, str) and not isinstance(columns, list):
            raise ValueError('columns doit être une chaîne ou une liste')
        if isinstance(columns, str) and columns != 'all':
            raise ValueError('columns doit être "all" ou une liste de noms de colonnes')
        if isinstance(columns, list):
            for col in columns:
                if col not in self._data.columns.tolist():
                    raise ValueError(f'Colonne inconnue : {col}')
        if not isinstance(legend, bool):
            raise ValueError('legend doit être un booléen')
        if not isinstance(bins, int) or bins < 1:
            raise ValueError('bins doit être un entier positif')
        self._columns = columns
        self._legend = legend
        self._bins = bins

    def vizs(self):
        """
        Génère et affiche l'histogramme.
        """
        if self._columns == 'all':
            cols = self._data.columns.tolist()
        else:
            cols = self._columns
        plt.figure(figsize=(8, 6))
        for col in cols:
            plt.hist(self._data[col], bins=self._bins, alpha=0.7, label=col, edgecolor='black')
        plt.xlabel('Valeur')
        plt.ylabel('Fréquence')
        plt.title('Histogramme')
        if self._legend and (len(cols) > 1):
            plt.legend()
        plt.tight_layout()
        self._figure = plt.gcf()
        self._auto_save()
        return self._figure
