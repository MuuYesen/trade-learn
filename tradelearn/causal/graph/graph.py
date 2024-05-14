import pandas as pd

from .causallearn.pc import PC
from .causallearn.ges import GES
from .causallearn.utils.GraphUtils import GraphUtils


class Graph:

    def __init__(self):
        pass

    @staticmethod
    def fit_causal(data: pd.DataFrame = None, method: str = 'pc', is_discrete: bool = True, filename: str = None):
        if method == 'pc':
            ict = "fisherz"
            if is_discrete:
                ict = "chisq"
            g_pred = PC(data.values, indep_test=ict).G

        if method == 'ges':
            g_pred = GES(data)['G']

        pdy = GraphUtils.to_pydot(g_pred)
        if filename is None:
            filename = method + '_graph.png'
        pdy.write_png(filename)
