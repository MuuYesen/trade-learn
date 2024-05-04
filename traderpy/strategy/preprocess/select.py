import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


class Select:

    def __init__(self):
        pass

    @staticmethod # 计算因子与收益率的IC值
    def ic_selection(factor_data, bt_x_test, returns, select=1, threshold=0.2, rank=20):
        ic_values={}
        for factor in factor_data.columns:
            print(factor_data[factor])
            factor_data[factor].astype('float')
            returns.astype('float')
            ic, _=pearsonr(factor_data[factor], returns)
            ic_values[factor]=ic

        # 将IC值转换为数值类型
        ic_values={factor: float(ic) for factor, ic in ic_values.items()}

        if select == 0:
            # 1-根据IC值设置阈值筛选因子
            selected_factors=[factor for factor, ic in ic_values.items() if abs(ic) > threshold]  # 选择IC绝对值大于0.2的因子
        if select == 1:
            # 2-筛选IC值前rank个的因子
            sorted_ic_values=sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True)
            selected_factors=[factor for factor, ic in sorted_ic_values[:rank]]  # 选择IC绝对值最大的前20个因子
        bt_x_train = factor_data[selected_factors]
        bt_x_test = bt_x_test[selected_factors]

        return bt_x_train,bt_x_test

    @staticmethod # 逐步回归筛选因子
    def stepwise_selection(X, bt_x_test, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
        included=list(initial_list)
        while True:
            changed=False
            # forward step
            excluded=list(set(X.columns) - set(included))
            new_pval=pd.Series(index=excluded)
            for new_column in excluded:
                model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column]=model.pvalues[new_column]
            best_pval=new_pval.min()
            if best_pval < threshold_in:
                best_feature=new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues=model.pvalues.iloc[1:]
            worst_pval=pvalues.max()  # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature=pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        bt_x_train=X[included]
        bt_x_test=bt_x_test[included]

        return bt_x_train, bt_x_test

    @staticmethod
    def pca_selection(bt_x_train, bt_x_test,component=20):
        pca=PCA(n_components=component)  # 假设降维到20个主成分
        pca.fit(bt_x_train)
        bt_x_train_pca=pca.transform(bt_x_train)
        bt_x_test_pca=pca.transform(bt_x_test)

        return bt_x_train_pca, bt_x_test_pca

    @staticmethod
    def rf_selection(bt_x_train, bt_x_test,bt_y_train,num=20):
        model = RandomForestClassifier()
        model.fit(bt_x_train,bt_y_train)

        # 获取特征重要性
        feature_importances = model.feature_importances_

        # 根据特征重要性排序，选择重要性较高的特征
        sorted_indices = feature_importances.argsort()[::-1]
        selected_features=sorted_indices[:num]  # 假设self.num_selected_features是要选择的特征数量

        # 根据选择的特征索引，筛选训练集和测试集的特征
        bt_x_train_selected = bt_x_train[:, selected_features]
        bt_x_test_selected = bt_x_test[:, selected_features]

        return bt_x_train_selected, bt_x_test_selected