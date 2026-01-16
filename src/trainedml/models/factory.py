"""
Factory pour instancier les modèles ML de trainedml.
Permet de récupérer un modèle par son nom (KNN, Logistic Regression, Random Forest).
"""

from trainedml.models.knn import KNNModel
from trainedml.models.logistic import LogisticModel
from trainedml.models.random_forest import RandomForestModel

def get_model(model_name):
    """
    Instancie et retourne un modèle ML de trainedml à partir de son nom.

    Cette fonction agit comme une factory centralisée : elle permet de récupérer
    n'importe quel modèle supporté par trainedml à partir d'un nom (anglais ou français).

    Parameters
    ----------
    model_name : str
        Nom du modèle ('KNN', 'Logistic Regression', 'Random Forest', etc.).
        Les variantes (anglais/français, abréviations) sont supportées.

    Returns
    -------
    model : instance de modèle ML
        Instance du modèle ML prêt à l'emploi (KNNModel, LogisticModel, RandomForestModel).

    Raises
    ------
    ValueError
        Si le nom du modèle n'est pas reconnu.

    Examples
    --------
    >>> from trainedml.models.factory import get_model
    >>> model = get_model('KNN')
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)

    >>> model = get_model('logistic regression')
    >>> model.fit(X, y)

    >>> model = get_model('forêt aléatoire')
    >>> print(model)
    """
    name = model_name.lower()
    if name in ['knn', 'k-nn', 'k nearest neighbors', 'k plus proches voisins']:
        return KNNModel()
    elif name in ['logistic', 'logistic regression', 'régression logistique']:
        return LogisticModel()
    elif name in ['random forest', 'forêt aléatoire', 'rf']:
        return RandomForestModel()
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
