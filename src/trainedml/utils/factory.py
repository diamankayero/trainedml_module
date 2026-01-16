
"""
Expose la fonction get_model pour l'import direct depuis trainedml.utils.factory.

Permet d'utiliser :
>>> from trainedml.utils.factory import get_model
>>> model = get_model('KNN')
>>> model.fit(X, y)
"""

from trainedml.models.factory import get_model

__all__ = ["get_model"]
