"""
Implémentation du modèle de régression logistique pour trainedml.
"""
from .base import BaseModel
from sklearn.linear_model import LogisticRegression


class LogisticModel(BaseModel):
    """
    Modèle de classification par régression logistique.
    
    Args:
        max_iter (int): Nombre maximum d'itérations (défaut: 200)
        **kwargs: Autres hyperparamètres passés à LogisticRegression
    """
    def __init__(self, max_iter=200, **kwargs):
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle de régression logistique."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la classe pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne la précision du modèle sur les données de test."""
        return self.model.score(X, y)
