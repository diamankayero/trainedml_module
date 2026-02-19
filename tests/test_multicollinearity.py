"""
Test unitaire du module MulticollinearityViz et de la fonction vif_summary.
"""
import unittest
import pandas as pd
from trainedml.viz.multicollinearity import MulticollinearityViz, vif_summary

class TestMulticollinearityViz(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [5, 3, 6, 2, 1]
        })

    def test_vif_summary(self):
        vif = vif_summary(self.df)
        self.assertIsInstance(vif, pd.Series)
        self.assertTrue(all(col in vif.index for col in self.df.columns))

    def test_multicollinearity_viz(self):
        viz = MulticollinearityViz(self.df)
        viz.vizs()
        self.assertIsNotNone(viz.figure)

if __name__ == '__main__':
    unittest.main()
