"""
Base class for trainedml visualizations.

This module provides the Vizs class, which serves as a base for all visualization classes
in trainedml. It defines the interface and common attributes for visualizations.

Examples
--------
>>> from trainedml.viz.vizs import Vizs
>>> class MyViz(Vizs):
...     def vizs(self):
...         # custom plotting code
...         pass
"""

import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


class Vizs(object):
    """
    Classe de base pour toutes les visualisations.
    Toutes les visualisations doivent hériter de cette classe et surcharger la méthode vizs().
    
    Attributes:
        _data: DataFrame pandas contenant les données
        _figure: Figure matplotlib générée
        _save_path: Chemin optionnel pour sauvegarder automatiquement la figure
    """
    def __init__(self, data, save_path: Optional[str] = None):
        """
        Initialise la visualisation.
        
        Args:
            data: DataFrame pandas contenant les données
            save_path (str, optional): Chemin pour sauvegarder la figure automatiquement.
                                       Formats supportés: png, pdf, svg, jpg, etc.
        """
        # Vérifie que les données sont bien un DataFrame pandas
        if not isinstance(data, pd.DataFrame):
            raise ValueError('data doit être un DataFrame pandas')
        self._data = data
        self._figure = None  # Stocke la figure générée (matplotlib, seaborn, etc.)
        self._save_path = save_path

    def vizs(self):
        """
        Méthode à surcharger dans les classes filles pour générer la visualisation.
        Appelle automatiquement save() si un save_path a été défini.
        """
        raise NotImplementedError('Les sous-classes doivent implémenter cette méthode')

    def save(self, path: Optional[str] = None, dpi: int = 150, **kwargs) -> Optional[str]:
        """
        Sauvegarde la figure dans un fichier.
        
        Args:
            path (str, optional): Chemin du fichier. Si None, utilise self._save_path
            dpi (int): Résolution de l'image (défaut: 150)
            **kwargs: Arguments supplémentaires passés à plt.savefig()
        
        Returns:
            str: Chemin du fichier sauvegardé, ou None si échec
        
        Raises:
            ValueError: Si aucun chemin n'est spécifié et _save_path est None
        """
        save_path = path or self._save_path
        
        if save_path is None:
            raise ValueError("Aucun chemin spécifié. Passez 'path' ou définissez save_path à l'initialisation.")
        
        # Créer le dossier parent si nécessaire
        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
            print(f"✅ Figure sauvegardée: {save_path}")
            return save_path
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None

    def _auto_save(self):
        """
        Sauvegarde automatique si un save_path a été défini.
        À appeler à la fin de vizs() dans les classes filles.
        """
        if self._save_path:
            self.save()

    @property
    def figure(self):
        """Retourne la figure générée."""
        return self._figure
    
    @property
    def save_path(self) -> Optional[str]:
        """Retourne le chemin de sauvegarde configuré."""
        return self._save_path
    
    @save_path.setter
    def save_path(self, value: Optional[str]):
        """Définit le chemin de sauvegarde."""
        self._save_path = value
