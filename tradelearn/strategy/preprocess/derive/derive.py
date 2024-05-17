import pandas as pd

from tradelearn.strategy.preprocess.derive.gplearn import genetic, fitness, functions


class Derive:

    def __init__(self):
        pass

    @staticmethod
    def generic_generate(data, f_col: list = None, n_alpha: int = 20):

        if f_col is None:
            try:
                f_col = data.columns.drop(['code', 'date', 'label'])
            except:
                f_col = data.columns.drop(['code', 'date'])

        function_set = functions._function_map.keys()

        est_gp = genetic.SymbolicTransformer(
            function_set=function_set,
            feature_names=f_col,
            metric='sharpe ratio',
            hall_of_fame=20,
            generations=3,
            population_size=100,
            n_components=n_alpha,
            parsimony_coefficient=0,
            tournament_size=40,
            init_depth=(2, 6),
            const_range=(-1, 1),
            p_crossover=0.6,
            p_subtree_mutation=0.01,
            p_hoist_mutation=0.05,
            p_point_mutation=0.01,
            p_point_replace=0.4,
            max_samples=0.9,
            low_memory=True,
            verbose=1,
            n_jobs=-1,
            init_method='grow'
        )

        c_return = data['close'].shift(-1).dropna()
        c_data = data[f_col].iloc[:-1]
        est_gp.fit(c_data, c_return)
        best_programs = est_gp._best_programs
        best_programs_dict = {}

        for p in best_programs:
            factor_name = 'alpha_' + str(best_programs.index(p) + 1) + '_' + str(round(p.fitness_,2))
            best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_, 'length': p.length_}

        best_programs = pd.DataFrame(best_programs_dict).T.sort_values(by='fitness')
        print(best_programs)

        gp_res = pd.DataFrame(est_gp.transform(data[f_col]), columns=best_programs_dict.keys())
        res_df = data.merge(gp_res, how='right', left_index=True, right_index=True)
        return res_df