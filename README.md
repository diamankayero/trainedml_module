

# trainedml

`trainedml` est un package Python qui fournit des outils simples pour **charger des jeux de données publics**, **entraîner et comparer des modèles de machine learning**, et **visualiser les résultats** de manière intuitive.

---

## 📦 Installation

### 🔹 Installation depuis **TestPyPI** (recommandé pour les tests)

```bash
python -m pip install --index-url https://test.pypi.org/simple/ trainedml --upgrade
```

> ⚠️ L’option `--no-deps` est nécessaire car TestPyPI ne contient pas toujours toutes les dépendances.

Si besoin, installe ensuite les dépendances séparément :

```bash
pip install -r requirements.txt
```

---

### 🔹 Installation classique (depuis les sources)

```bash
git clone https://github.com/diamankayero/trainedml.git
cd trainedml
pip install -r requirements.txt
```

---

## 🚀 Fonctionnalités principales

* Chargement de jeux de données publics (ex : **Iris**)
* Modèles implémentés :

  * KNN
  * Régression logistique
  * Random Forest
* Visualisations :

  * Heatmap
  * Histogrammes
  * Courbes
* API simple pour :

  * l’entraînement
  * l’évaluation
  * la comparaison de modèles

---

## 🧪 Exemple d’utilisation

```python
from trainedml.data.loader import DataLoader
from trainedml.models.knn import KNNModel
from trainedml.visualization import Visualizer

# Chargement des données
iris = DataLoader().load_iris()

# Entraînement d'un modèle
X = iris.drop(columns=['species'])
y = iris['species']

model = KNNModel()
model.fit(X, y)

# Visualisation
viz = Visualizer(iris)
fig = viz.heatmap()
fig.show()
```

---

## ✅ Tests

Pour exécuter les tests unitaires :

```bash
python -m unittest discover tests
```

---

## 📌 Statut du projet

* ✔️ Version de test publiée sur **TestPyPI**
* 🔄 En développement actif
* 🚀 Publication sur PyPI prévue après validation
# trainedml_module
