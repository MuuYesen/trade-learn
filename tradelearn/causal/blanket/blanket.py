from .pyfeature.iamb import IAMB
from .pyfeature.pcmb import PCMB


class Blanket:

    def __init__(self):
        pass

    @staticmethod
    def fit_causal(data=None, method='iamb', target_name=None, alpha=0.05, is_discrete=True):
        target = data.columns.tolist().index(target_name)

        res_id = None
        if method == 'iamb':
            res_id = IAMB(data, target, alpha, is_discrete)
        if method == 'pcmb':
            res_id = PCMB(data, target, alpha, is_discrete)

        res = data.iloc[:, res_id]
        return res
