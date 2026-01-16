"""
Data analysis and exploratory statistics for trainedml.

This module provides the DataAnalyzer class, which offers a suite of methods for
descriptive statistics, distribution analysis, correlation, missing values, outliers,
target analysis, boxplots, bivariate analysis, normality, multicollinearity, and profiling.

Mathematical context
--------------------
- Correlation: Pearson, Spearman, Kendall
- Outlier detection: IQR, Z-score
- Normality: Shapiro-Wilk, D'Agostino, Anderson-Darling
- Multicollinearity: Variance Inflation Factor (VIF)

Examples
--------
>>> from trainedml.analyzer import DataAnalyzer
>>> analyzer = DataAnalyzer(df)
>>> analyzer.correlation()
>>> analyzer.outliers()
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataAnalyzer:
    r"""
    Exploratory data analysis and statistics.

    Provides a suite of methods for descriptive statistics, distribution analysis, correlation,
    missing values, outliers, target analysis, boxplots, bivariate analysis, normality,
    multicollinearity, and profiling.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze.

    Attributes
    ----------
    data : pandas.DataFrame
        The underlying data.

    Examples
    --------
    Basic usage:
    >>> from trainedml.analyzer import DataAnalyzer
    >>> analyzer = DataAnalyzer(df)
    >>> stats = analyzer.distribution()
    >>> print(stats)

    Correlation matrix:
    >>> corr = analyzer.correlation(method='spearman')
    >>> print(corr)

    Outlier detection:
    >>> out = analyzer.outliers(method='zscore', threshold=3)
    >>> print(out)

    Normality tests:
    >>> norm = analyzer.normality()
    >>> print(norm)

    Profiling report:
    >>> report = analyzer.profiling()
    >>> print(report['describe'])

    Notes
    -----
    - All methods return pandas objects or dicts for easy integration with pandas workflows.
    - For plotting, returned objects are matplotlib figures.
    """
    def __init__(self, data):
        self.data = data

    def distribution(self, columns='all', **kwargs):
        """
        Compute and plot the distribution of variables.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to analyze.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        dict
            Summary statistics and figures.

        Examples
        --------
        >>> stats = analyzer.distribution()
        >>> print(stats)
        >>> stats = analyzer.distribution(columns=['A', 'B'])
        >>> print(stats)
        """
        # ...existing code...

    def correlation(self, features='all', method='pearson', mask=True, **kwargs):
        """
        Compute the correlation matrix between features.

        Parameters
        ----------
        features : 'all' or list, default='all'
            Features to include.
        method : str, default='pearson'
            Correlation method ('pearson', 'spearman', 'kendall').
        mask : bool, default=True
            Whether to mask the upper triangle.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.

        Notes
        -----
        Pearson correlation:
        $r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$

        Examples
        --------
        >>> corr = analyzer.correlation()
        >>> print(corr)
        >>> corr = analyzer.correlation(features=['A', 'B'], method='kendall')
        >>> print(corr)
        """
        # ...existing code...

    def missing(self, **kwargs):
        """
        Analyze missing values in the dataset.

        Returns
        -------
        pandas.Series
            Count of missing values per column.

        Examples
        --------
        >>> missing = analyzer.missing()
        >>> print(missing)
        """
        # ...existing code...

    def outliers(self, method='iqr', threshold=1.5, **kwargs):
        """
        Detect outliers in the dataset.

        Parameters
        ----------
        method : str, default='iqr'
            Outlier detection method ('iqr', 'zscore').
        threshold : float, default=1.5
            Threshold for outlier detection.
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Outlier summary per column.

        Notes
        -----
        IQR method:
        $Q_1 = 25\%$ percentile, $Q_3 = 75\%$ percentile
        $IQR = Q_3 - Q_1$
        Outlier if $x < Q_1 - k \cdot IQR$ or $x > Q_3 + k \cdot IQR$

        Examples
        --------
        >>> out = analyzer.outliers()
        >>> print(out)
        >>> out = analyzer.outliers(method='zscore', threshold=3)
        >>> print(out)
        """
        # ...existing code...

    def target(self, target_column, **kwargs):
        """
        Analyze the target variable (distribution, imbalance, etc.).

        Parameters
        ----------
        target_column : str
            Name of the target column.
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Target analysis summary.

        Examples
        --------
        >>> target = analyzer.target(target_column='species')
        >>> print(target)
        """
        # ...existing code...

    def boxplot(self, columns='all', by=None, **kwargs):
        """
        Generate boxplots for selected columns.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to plot.
        by : str or None, default=None
            Grouping variable.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot figure.

        Examples
        --------
        >>> fig = analyzer.boxplot(columns=['A', 'B'], by='Group')
        >>> fig.show()
        """
        # ...existing code...

    def bivariate(self, x, y, **kwargs):
        """
        Bivariate analysis between two variables (scatter, etc.).

        Parameters
        ----------
        x : str
            First variable.
        y : str
            Second variable.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The generated bivariate plot.

        Examples
        --------
        >>> fig = analyzer.bivariate(x='A', y='B')
        >>> fig.show()
        """
        # ...existing code...

    def normality(self, columns='all', **kwargs):
        """
        Test normality of variables (Shapiro, D'Agostino, Anderson).

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to test.
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Normality test results per column.

        Examples
        --------
        >>> norm = analyzer.normality()
        >>> print(norm)
        """
        # ...existing code...

    def multicollinearity(self, **kwargs):
        """
        Analyze multicollinearity using Variance Inflation Factor (VIF).

        Returns
        -------
        pandas.Series
            VIF per feature.

        Notes
        -----
        $VIF_j = \frac{1}{1 - R_j^2}$
        where $R_j^2$ is the $R^2$ of regressing feature $j$ on all others.

        Examples
        --------
        >>> vif = analyzer.multicollinearity()
        >>> print(vif)
        """
        # ...existing code...

    def profiling(self, **kwargs):
        """
        Generate a global profiling report (summary statistics, missing, outliers, etc.).

        Returns
        -------
        dict
            Profiling report.

        Examples
        --------
        >>> report = analyzer.profiling()
        >>> print(report['describe'])
        """
        # ...existing code...
