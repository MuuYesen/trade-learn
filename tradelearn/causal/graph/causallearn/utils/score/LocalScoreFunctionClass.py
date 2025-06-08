from typing import Any, Callable, Dict, List

from .LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from .ScoreUtils import *


class LocalScoreClass(object):
    def __init__(
        self,
        data: Any,
        local_score_fun: Callable[[Any, int, List[int], Any], float],
        parameters=None,
    ):
        self.data = data
        self.local_score_fun = local_score_fun
        self.parameters = parameters
        self.score_cache = {}

        if self.local_score_fun == local_score_BIC_from_cov:
            self.cov = np.cov(self.data.T)
            self.n = self.data.shape[0]

    def score(self, i: int, PAi: List[int]) -> float:
        if i not in self.score_cache:
            self.score_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            if self.local_score_fun == local_score_BIC_from_cov:
                self.score_cache[i][hash_key] = self.local_score_fun((self.cov, self.n), i, PAi, self.parameters)
            else:
                self.score_cache[i][hash_key] = self.local_score_fun(self.data, i, PAi, self.parameters)

        return self.score_cache[i][hash_key]
