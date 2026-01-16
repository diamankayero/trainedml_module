
"""
Command-line interface (CLI) for trainedml.

This script provides a simple and flexible CLI for running machine learning pipelines
with trainedml: data loading, model training, evaluation, benchmarking, and visualization.

Features
--------
- Load public datasets or remote CSVs
- Train/test split with configurable seed and test size
- Automatic task type detection (classification vs regression)
- Model selection (KNN, Logistic Regression, Random Forest, ...)
- Benchmarking of all models for the task
- Visualization: heatmap, histogram, line plot
- Output of evaluation metrics and timings

Examples (to run in terminal)
----------------------------
Entrainer un modèle Random Forest sur Iris et afficher la heatmap :
    python -m trainedml.cli --model random_forest --dataset iris --show

Comparer tous les modèles sur Wine (benchmark) :
    python -m trainedml.cli --dataset wine --benchmark --show

Charger un CSV distant et tracer une courbe :
    python -m trainedml.cli --url https://.../data.csv --target classe --line feature1 feature2 --show

Afficher un histogramme des colonnes numériques :
    python -m trainedml.cli --dataset iris --histogram --show

Notes
-----
- Utilisez --show pour afficher les figures matplotlib à la fin du script.
- Le CLI détecte automatiquement le type de tâche (classification/régression).
"""

import argparse
from trainedml.data.loader import DataLoader
from trainedml.models import MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP, get_model
from trainedml.evaluation import Evaluator
from trainedml.visualization import Visualizer
from sklearn.model_selection import train_test_split


def _is_classification_target(y):
    """
    Détermine si la cible est catégorielle (classification) ou numérique (régression).

    Parameters
    ----------
    y : pandas.Series
        Colonne cible à analyser.

    Returns
    -------
    bool
        True si classification, False si régression.

    Examples
    --------
    >>> _is_classification_target(df['species'])
    True
    >>> _is_classification_target(df['target'])
    False
    """
    import pandas as pd
    import numpy as np
    # Si c'est du texte ou catégoriel, c'est de la classification
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        return True
    # Si peu de valeurs uniques (<= 20) et entiers, probablement classification
    if len(y.unique()) <= 20 and np.issubdtype(y.dtype, np.integer):
        return True
    return False



def main():

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="trainedml: pipeline ML simple")
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='random_forest', help='Type de modèle à utiliser')
    parser.add_argument('--dataset', type=str, default='iris', help='Nom du dataset (iris, wine)')
    parser.add_argument('--url', type=str, default=None, help='URL d\'un CSV distant')
    parser.add_argument('--target', type=str, default=None, help='Nom de la colonne cible (si url)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le split train/test')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proportion de test (0-1)')
    parser.add_argument('--show', action='store_true', help='Afficher la heatmap après entraînement')
    parser.add_argument('--histogram', action='store_true', help='Afficher un histogramme des colonnes numériques')
    parser.add_argument('--benchmark', action='store_true', help='Comparer tous les modèles et afficher scores et temps')
    parser.add_argument('--line', nargs=2, metavar=('X', 'Y'), help='Tracer une courbe (line plot) entre deux colonnes')
    args = parser.parse_args()


    # --- Data loading ---
    print(f"Chargement du dataset {args.dataset if args.url is None else args.url} ...")
    loader = DataLoader()
    X, y = loader.load_dataset(name=args.dataset if args.url is None else None, url=args.url, target=args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Taille X_train : {X_train.shape}, X_test : {X_test.shape} (seed={args.seed})")


    # --- Task type detection ---
    is_classification = _is_classification_target(y)
    task_type = "classification" if is_classification else "régression"
    print(f"Type de tâche détecté : {task_type}")


    # --- DataFrame for visualization ---
    import pandas as pd
    if args.url is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = loader.load_csv_from_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") if args.dataset == "iris" else pd.concat([X, y], axis=1)

    viz = Visualizer(data)
    numeric_cols = [col for col in data.columns if data[col].dtype != 'O']

    # --- Benchmark mode ---
    if args.benchmark:
        print("\n--- BENCHMARK ---")
        from trainedml.benchmark import Benchmark
        # Utiliser uniquement les modèles adaptés au type de tâche
        if is_classification:
            models_to_use = CLASSIFIER_MAP
            print(f"Utilisation des classificateurs : {list(models_to_use.keys())}")
        else:
            models_to_use = REGRESSOR_MAP
            print(f"Utilisation des régresseurs : {list(models_to_use.keys())}")
        
        models = {name: cls() for name, cls in models_to_use.items()}
        bench = Benchmark(models)
        results = bench.run(X_train, y_train, X_test, y_test)
        for name, res in results.items():
            print(f"\nModèle : {name}")
            for metric, value in res['scores'].items():
                print(f"  {metric}: {value:.3f}")
            print(f"  fit_time: {res['fit_time']:.4f} s")
            print(f"  predict_time: {res['predict_time']:.4f} s")

    # --- Single model mode ---
    else:
        print(f"Entraînement du modèle {args.model}...")
        model = MODEL_MAP[args.model]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Évaluation :")
        scores = Evaluator.evaluate_all(y_test, y_pred)
        for metric, value in scores.items():
            print(f"{metric}: {value:.3f}")


    # --- Visualization options ---
    if args.line:
        x_col, y_col = args.line
        print(f"Génération de la courbe {y_col} en fonction de {x_col}...")
        fig = viz.line(x_column=x_col, y_column=y_col)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la courbe.")
    elif args.histogram:
        print("Génération de l'histogramme des colonnes numériques...")
        fig = viz.histogram(columns=numeric_cols, legend=True)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher l'histogramme.")
    else:
        print("Génération de la heatmap de corrélation...")
        fig = viz.heatmap(features=numeric_cols)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la heatmap.")

if __name__ == "__main__":
    main()
