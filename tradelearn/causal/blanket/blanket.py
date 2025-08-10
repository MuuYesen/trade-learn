import pandas as pd

from .pyfeature.iamb import IAMB
from .pyfeature.pcmb import PCMB


class Blanket:
    """A class to implement the Blanket algorithm for causal inference."""


    def __init__(self):
        """Initializes the Blanket class."""
        pass

    @staticmethod
    def fit_causal(data: pd.DataFrame = None, method: str = 'iamb', target: str = None,
                   alpha: float = 0.05, is_discrete: bool = True):
        """Fits the causal model using the specified method.

        Args:
            data (pd.DataFrame): The input data for causal inference.
            method (str): The method to use for causal inference ('iamb' or 'pcmb').
            target (str): The target variable for causal inference.
            alpha (float): The significance level for the tests.
            is_discrete (bool): Indicates if the data is discrete.

        Returns:
            pd.DataFrame: The resulting DataFrame containing the selected features.
        """

        target = data.columns.tolist().index(target)  # Get the index of the target variable.

        res_id = None  # Initialize the result ID.
        if method == 'iamb':
            res_id = IAMB(data, target, alpha, is_discrete)  # Use IAMB method for causal inference.
        if method == 'pcmb':
            res_id = PCMB(data, target, alpha, is_discrete)  # Use PCMB method for causal inference.

        res = data.iloc[:, res_id]  # Select the resulting features based on the method used.
        return res  # Return the resulting DataFrame.
