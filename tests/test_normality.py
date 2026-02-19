"""
Test unitaire du module NormalityViz et de la fonction normality_tests.
"""
import unittest
import pandas as pd
from trainedml.viz.normality import NormalityViz, normality_tests

class TestNormalityViz(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [2.0, 2.1, 2.2, 2.3, 2.4]
        })

    def test_normality_tests(self):
        results = normality_tests(self.df)
        self.assertIsInstance(results, dict)
        self.assertIn('A', results)
        self.assertIn('B', results)

    def test_normality_viz(self):
        viz = NormalityViz(self.df)
        viz.vizs()
        self.assertIsNotNone(viz.figure)

if __name__ == '__main__':
    unittest.main()
