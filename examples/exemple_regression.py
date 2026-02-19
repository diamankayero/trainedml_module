# Exemples avancés : pipeline régression

```python
import pandas as pd
from trainedml.data.loader import DataLoader
from trainedml.models.regressors import LinearRegressorModel, RandomForestRegressorModel
from trainedml.viz.profiling import profiling_report
from trainedml.viz.line import LineViz

# Chargement d'un dataset public (exemple : Boston Housing)
# Remplacer par un CSV ou un dataset compatible
# X, y = DataLoader().load_dataset(name="boston")
# Pour l'exemple, on crée un DataFrame fictif :
X = pd.DataFrame({
    'surface': [30, 45, 60, 80, 100],
    'pieces': [1, 2, 3, 4, 5]
})
y = pd.Series([100, 150, 200, 250, 300])

# Profiling rapide
desc = profiling_report(X)
print(desc['summary'])

# Entraînement d'un modèle de régression linéaire
model = LinearRegressorModel()
model.fit(X, y)
preds = model.predict(X)
print("Prédictions:", preds)

# Visualisation de la relation surface/prix
viz = LineViz(X.assign(prix=y), x_column='surface', y_column='prix')
viz.vizs()
viz.figure.show()
```
