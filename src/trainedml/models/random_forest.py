"""
Implémentation du modèle Random Forest pour trainedml.
"""
from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    """
    Modèle de classification Random Forest.
    
    Args:
        n_estimators (int): Nombre d'arbres dans la forêt (défaut: 100)
        **kwargs: Autres hyperparamètres passés à RandomForestClassifier
    """
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle Random Forest."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la classe pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne la précision du modèle sur les données de test."""
        return self.model.score(X, y)
