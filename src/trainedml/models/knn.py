"""
Implémentation du modèle K-Nearest Neighbors pour trainedml.
"""
from .base import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(BaseModel):
    """
    Modèle de classification K-Nearest Neighbors.
    
    Args:
        n_neighbors (int): Nombre de voisins à considérer (défaut: 5)
        **kwargs: Autres hyperparamètres passés à KNeighborsClassifier
    """
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle KNN."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la classe pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne la précision du modèle sur les données de test."""
        return self.model.score(X, y)
