"""
Implémentation des modèles de régression pour trainedml.
Contient les régresseurs : RandomForestRegressor, KNNRegressor, LinearRegressor.
"""
from .base import BaseRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class RandomForestRegressorModel(BaseRegressor):
    """
    Modèle de régression Random Forest.
    
    Args:
        n_estimators (int): Nombre d'arbres dans la forêt (défaut: 100)
        **kwargs: Autres hyperparamètres passés à RandomForestRegressor
    """
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle Random Forest."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la valeur cible pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne le score R² du modèle sur les données de test."""
        return self.model.score(X, y)


class KNNRegressorModel(BaseRegressor):
    """
    Modèle de régression K-Nearest Neighbors.
    
    Args:
        n_neighbors (int): Nombre de voisins à considérer (défaut: 5)
        **kwargs: Autres hyperparamètres passés à KNeighborsRegressor
    """
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__()
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle KNN."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la valeur cible pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne le score R² du modèle sur les données de test."""
        return self.model.score(X, y)


class LinearRegressorModel(BaseRegressor):
    """
    Modèle de régression linéaire.
    
    Args:
        **kwargs: Hyperparamètres passés à LinearRegression
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Entraîne le modèle de régression linéaire."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la valeur cible pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne le score R² du modèle sur les données de test."""
        return self.model.score(X, y)


class RidgeRegressorModel(BaseRegressor):
    """
    Modèle de régression Ridge (L2).
    
    Args:
        alpha (float): Paramètre de régularisation (défaut: 1.0)
        **kwargs: Autres hyperparamètres passés à Ridge
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle Ridge."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la valeur cible pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne le score R² du modèle sur les données de test."""
        return self.model.score(X, y)


class LassoRegressorModel(BaseRegressor):
    """
    Modèle de régression Lasso (L1).
    
    Args:
        alpha (float): Paramètre de régularisation (défaut: 1.0)
        **kwargs: Autres hyperparamètres passés à Lasso
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)

    def fit(self, X, y):
        """Entraîne le modèle Lasso."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la valeur cible pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne le score R² du modèle sur les données de test."""
        return self.model.score(X, y)
