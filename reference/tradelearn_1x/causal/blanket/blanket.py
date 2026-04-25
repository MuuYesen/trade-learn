import pandas as pd

from .pyfeature.iamb import IAMB
from .pyfeature.pcmb import PCMB


class Blanket:

    def __init__(self):
        pass

    @staticmethod
    def fit_causal(data: pd.DataFrame = None, method: str = 'iamb', target: str = None,
                   alpha: float = 0.05, is_discrete: bool = True):

        target = data.columns.tolist().index(target)

        res_id = None
        if method == 'iamb':
            res_id = IAMB(data, target, alpha, is_discrete)
        if method == 'pcmb':
            res_id = PCMB(data, target, alpha, is_discrete)

        res = data.iloc[:, res_id]
        return res
