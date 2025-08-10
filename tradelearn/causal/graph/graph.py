import pandas as pd

from .causallearn.pc import PC
from .causallearn.ges import GES
from .causallearn.utils.GraphUtils import GraphUtils

class Graph:
    """A class to represent and manipulate causal graphs."""


    def __init__(self):
        """Initializes the Graph class."""
        pass

    @staticmethod
    def fit_causal(data: pd.DataFrame = None, method: str = 'pc', is_discrete: bool = True, filename: str = None):
        """Fits a causal graph using the specified method and saves it as an image.

        Args:
            data (pd.DataFrame): The input data for causal inference.
            method (str): The method to use for causal inference ('pc' or 'ges').
            is_discrete (bool): Indicates if the data is discrete.
            filename (str): The filename to save the graph image. If None, defaults to '{method}_graph.png'.

        Returns:
            None
        """
        if method == 'pc':
            ict = "fisherz"  # Default independence test for continuous data.
            if is_discrete:
                ict = "chisq"  # Use chi-squared test for discrete data.
            g_pred = PC(data.values, indep_test=ict).G  # Fit the PC algorithm to the data.

        if method == 'ges':
            g_pred = GES(data)['G']  # Fit the GES algorithm to the data.

        pdy = GraphUtils.to_pydot(g_pred)  # Convert the graph to a PyDot object.
        if filename is None:
            filename = method + '_graph.png'  # Default filename for the graph image.
        pdy.write_png(filename)  # Save the graph as a PNG image.
