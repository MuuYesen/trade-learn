import numpy as np
import pandas as pd

from tradelearn.strategy.preprocess.derive.gplearn import genetic, fitness, functions

import random

class Derive:

    def __init__(self):
        pass

    @staticmethod
    def generic_generate(dataX, dataY):
        def _genBetterFormula(fitResult, threshold):
            '''
            输入fit后的est_gp和fitness的阈值，从最后一代选出超过阈值的公式
            '''
            if fitResult.metric.greater_is_better:
                better_fitness = [f.__str__() for f in fitResult._programs[-1] if f.fitness_ > threshold]
            else:
                better_fitness = [f.__str__() for f in fitResult._programs[-1] if f.fitness_ < threshold]
            return better_fitness

        def score_func_basic(y, y_pred, sample_weight):  # 因子评价指标
            if len(np.unique(y_pred[-1])) <= 10:  # 没办法分组
                ic = -1
            else:
                corr_df = pd.DataFrame(y).corrwith(pd.DataFrame(y_pred), axis=1, method='spearman')
                ic = abs(corr_df.mean())
            return ic if not np.isnan(ic) else 0  # pearson

        # 函数集
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                        'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 'max', 'min'
                        ]

        feature_names = dataX.columns

        est_gp = genetic.SymbolicTransformer(
            hall_of_fame=20,
            population_size=30,
            generations=10,
            tournament_size=10,
            init_depth=(1, 3),
            stopping_criteria=500,
            max_samples=0.9,
            low_memory=True,
            feature_names=feature_names,
            function_set=function_set,
            metric=fitness.make_fitness(score_func_basic, greater_is_better=True),
            verbose=1,
            const_range=(3, 7),
            init_method='grow'
        )

        # y = data['return']
        # X = data.drop(['return'], axis=1)

        est_gp.fit(dataX, dataY)

        for program in est_gp:  # 筛选的最优公式集，相同的公式会对应生成三个相同的特征
            print(program)
            print(program.raw_fitness_, '\n')

        return est_gp.transform(dataX)
        # #获取符合要求的goodAlpha
        # alpha = _genBetterFormula(est_gp, 0.01)
        #
        # gp_features = alpha.transform(dataX)
        #
        # return gp_features
