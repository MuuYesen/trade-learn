from traderpy.strategy.preprocess.derive.libs.gplearn import genetic, fitness, functions


class Derive:

    def __init__(self):
        pass

    @staticmethod
    def generic_generate():
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
                # print(len(np.unique(y_pred)))
                ic = -1
            else:
                corr_df = pd.DataFrame(y).corrwith(pd.DataFrame(y_pred), axis=1, method='spearman')
                ic = abs(corr_df.mean())
            return ic if not np.isnan(ic) else 0  # pearson

        est_gp = genetic.SymbolicRegressor(
            population_size=30,
            generations=2,
            tournament_size=10,
            init_depth=(1, 3),
            stopping_criteria=500,
            max_samples=0.9,
            low_memory=True,
            feature_names=feature_names,
            function_set=function_set,
            metric=fitness.make_fitness(score_func_basic),
            verbose=1,
            const_range=(3, 7),
            init_method='grow'
        )

        est_gp.fit(X, y)
        #获取符合要求的goodAlpha
        alpha = _genBetterFormula(est_gp, 0.01)
        # 去重
        alpha = list(set(alpha))
        gp_features = alpha.transform(X)
        return gp_features

    @staticmethod
    def compose_generate():
        pass
