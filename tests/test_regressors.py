"""
Test unitaire des modèles de régression (RandomForest, KNN, Linear, Ridge, Lasso).
"""
import unittest
import pandas as pd
from trainedml.models.regressors import (
    RandomForestRegressorModel, KNNRegressorModel, LinearRegressorModel, RidgeRegressorModel, LassoRegressorModel
)

class TestRegressors(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 4, 6, 8, 10]
        })
        self.y = pd.Series([1.1, 2.2, 3.0, 4.1, 5.2])

    def test_random_forest(self):
        model = RandomForestRegressorModel(n_estimators=10)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        score = model.evaluate(self.X, self.y)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_knn(self):
        model = KNNRegressorModel(n_neighbors=2)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        score = model.evaluate(self.X, self.y)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_linear(self):
        model = LinearRegressorModel()
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        score = model.evaluate(self.X, self.y)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_ridge(self):
        model = RidgeRegressorModel(alpha=1.0)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        score = model.evaluate(self.X, self.y)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_lasso(self):
        model = LassoRegressorModel(alpha=0.1)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        score = model.evaluate(self.X, self.y)
        self.assertTrue(-1.0 <= score <= 1.0)

if __name__ == '__main__':
    unittest.main()
