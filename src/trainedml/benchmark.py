"""
Benchmark utilities for comparing multiple models in trainedml.

This module provides the Benchmark class to compare the performance (accuracy, speed, etc.)
of several models on the same dataset, with optional parallelization and progress bar.

Mathematical Formulation
------------------------
Let $\mathcal{M} = \{M_1, ..., M_K\}$ be a set of models. For each model $M_k$:
- Fit time: $T_{fit}^{(k)}$
- Predict time: $T_{pred}^{(k)}$
- Score: $S^{(k)}$ (e.g., accuracy)

The benchmark returns a dictionary:

.. code-block:: python

    {
        'model_name': {
            'scores': {...},
            'fit_time': ...,
            'predict_time': ...
        },
        ...
    }

Examples
--------
>>> from trainedml.benchmark import Benchmark
>>> models = {'knn': KNNModel(), 'rf': RandomForestModel()}
>>> bench = Benchmark(models)
>>> results = bench.run(X_train, y_train, X_test, y_test)
>>> print(results)
"""

import time
from typing import Dict, Any, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
from .evaluation import Evaluator


def _train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    Helper function to train and evaluate a single model (for parallelization).

    Parameters
    ----------
    name : str
        Model name.
    model : object
        Model instance (must implement fit, predict).
    X_train, y_train, X_test, y_test : array-like
        Data splits.

    Returns
    -------
    tuple
        (model name, results dict)
    """
    # Mesure du temps d'entraînement
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Mesure du temps de prédiction
    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred

    scores = Evaluator.evaluate_all(y_test, y_pred)
    return name, {
        'scores': scores,
        'fit_time': fit_time,
        'predict_time': predict_time
    }


class Benchmark:
    r"""
    Class for comparing the performance of multiple classification/regression models.

    Supports sequential or parallel execution, progress bar, and timing.

    Parameters
    ----------
    models : dict
        Dictionary {name: model_instance}.

    Attributes
    ----------
    models : dict
        Models to benchmark.
    results : dict or None
        Results after running the benchmark.

    Methods
    -------
    run(X_train, y_train, X_test, y_test, parallel=False, n_jobs=-1, show_progress=True)
        Run the benchmark and return results.
    summary()
        Return a formatted summary of the results.
    print_summary()
        Print the summary to stdout.

    Examples
    --------
    >>> bench = Benchmark({'knn': KNNModel(), 'rf': RandomForestModel()})
    >>> results = bench.run(X_train, y_train, X_test, y_test)
    >>> bench.print_summary()
    """
    def __init__(self, models: Dict[str, Any]):
        """
        Args:
            models (dict): dictionnaire {nom: instance_modele}
        """
        self.models = models
        self.results = None

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        parallel: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Train and evaluate each model, returning scores and timing.

        Parameters
        ----------
        X_train, y_train, X_test, y_test : array-like
            Data splits.
        parallel : bool, default=False
            If True, run models in parallel.
        n_jobs : int, default=-1
            Number of jobs for parallelization.
        show_progress : bool, default=True
            Show a progress bar.

        Returns
        -------
        dict
            {model_name: {scores, fit_time, predict_time}}
        """
        results = {}
        
        if parallel:
            # Exécution parallèle avec joblib
            model_items = list(self.models.items())
            
            if show_progress:
                print(f"🚀 Benchmark parallèle de {len(model_items)} modèles...")
            
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_train_and_evaluate)(
                    name, model, X_train, y_train, X_test, y_test
                )
                for name, model in tqdm(
                    model_items,
                    desc="Entraînement",
                    disable=not show_progress
                )
            )
            
            for name, res in parallel_results:
                results[name] = res
        else:
            # Exécution séquentielle avec barre de progression
            iterator = self.models.items()
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(self.models),
                    desc="Benchmark",
                    unit="modèle"
                )
            
            for name, model in iterator:
                if show_progress:
                    iterator.set_postfix({"modèle": name})
                
                # Mesure du temps d'entraînement
                start_fit = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - start_fit

                # Mesure du temps de prédiction
                start_pred = time.time()
                y_pred = model.predict(X_test)
                predict_time = time.time() - start_pred

                scores = Evaluator.evaluate_all(y_test, y_pred)
                results[name] = {
                    'scores': scores,
                    'fit_time': fit_time,
                    'predict_time': predict_time
                }
        
        self.results = results
        return results
    
    def summary(self) -> Optional[str]:
        """
        Return a formatted summary of the benchmark results.

        Returns
        -------
        str or None
            Text summary, or None if no results.
        """
        if self.results is None:
            return None
        
        lines = ["=" * 60, "📊 RÉSUMÉ DU BENCHMARK", "=" * 60]
        
        # Trouver le meilleur modèle par accuracy
        best_model = None
        best_accuracy = -1
        
        for name, res in self.results.items():
            lines.append(f"\n🔹 {name}")
            lines.append("-" * 40)
            for metric, value in res['scores'].items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append(f"  ⏱️ fit_time: {res['fit_time']:.4f}s")
            lines.append(f"  ⏱️ predict_time: {res['predict_time']:.4f}s")
            
            if res['scores'].get('accuracy', 0) > best_accuracy:
                best_accuracy = res['scores'].get('accuracy', 0)
                best_model = name
        
        if best_model:
            lines.append("\n" + "=" * 60)
            lines.append(f"🏆 MEILLEUR MODÈLE: {best_model} (accuracy: {best_accuracy:.4f})")
            lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def print_summary(self):
        """
        Print the summary of the benchmark results.
        """
        summary = self.summary()
        if summary:
            print(summary)
        else:
            print("⚠️ Aucun résultat. Exécutez d'abord run().")
