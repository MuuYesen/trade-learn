from .common.iamb import IAMB
from .common.pcmb import PCMB


class Blanket:

    def __init__(self):
        pass

    @staticmethod
    def fit_causal(data=None, method='iamb', target=0, alpha=0.05, is_discrete=True):
        res_id = None
        if method == 'iamb':
            res_id = IAMB(data, target, alpha, is_discrete)
        if method == 'pcmb':
            res_id = PCMB(data, target, alpha, is_discrete)

        res = data.iloc[:, res_id]
        return res
