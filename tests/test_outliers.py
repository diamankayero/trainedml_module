"""
Test unitaire du module OutliersViz et de la fonction outlier_summary.
"""
import unittest
import pandas as pd
from trainedml.viz.outliers import OutliersViz, outlier_summary

class TestOutliersViz(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 100, 5],
            'B': [5, 4, 3, 2, 1]
        })

    def test_outlier_summary(self):
        summary = outlier_summary(self.df, method='iqr')
        self.assertIsInstance(summary, dict)
        self.assertIn('A', summary)

    def test_outliers_viz(self):
        viz = OutliersViz(self.df)
        viz.vizs()
        self.assertIsNotNone(viz.figure)

if __name__ == '__main__':
    unittest.main()
