"""
Module de benchmark pour comparer plusieurs modèles sur un même jeu de données.
Supporte la parallélisation et l'affichage de la progression.
"""
import time
from typing import Dict, Any, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
from .evaluation import Evaluator


def _train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    Fonction helper pour entraîner et évaluer un modèle unique.
    Utilisée pour la parallélisation.
    
    Returns:
        tuple: (nom, résultats)
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
    """
    Classe pour comparer les performances de plusieurs modèles de classification/régression.
    
    Supporte:
    - Exécution séquentielle ou parallèle (via joblib)
    - Barre de progression (via tqdm)
    - Mesure des temps d'entraînement et de prédiction
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
        Entraîne et évalue chaque modèle, retourne les scores et les temps d'exécution.
        
        Args:
            X_train: Données d'entraînement
            y_train: Cibles d'entraînement
            X_test: Données de test
            y_test: Cibles de test
            parallel (bool): Si True, exécute les modèles en parallèle (défaut: False)
            n_jobs (int): Nombre de jobs pour la parallélisation (-1 = tous les cœurs)
            show_progress (bool): Si True, affiche une barre de progression (défaut: True)
        
        Returns:
            dict: {nom_modele: {scores, fit_time, predict_time}}
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
        Retourne un résumé formaté des résultats du benchmark.
        
        Returns:
            str: Résumé textuel des résultats, ou None si pas de résultats
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
        """Affiche le résumé du benchmark."""
        summary = self.summary()
        if summary:
            print(summary)
        else:
            print("⚠️ Aucun résultat. Exécutez d'abord run().")
