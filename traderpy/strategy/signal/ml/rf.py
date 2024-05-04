from traderpy.strategy.signal.tuner.StandardTuner import StandardTuner

import pandas as pd

# from sklearnex import patch_sklearn
# patch_sklearn()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RandomForestTiming:

    def __init__(self, **kwargs):
        self.clf = RandomForestClassifier(**kwargs)

    def fit(self, x_train, y_train, params={}):
        if params:
            self.clf.fit(x_train, y_train)
        else:
            self.clf = StandardTuner.tune_param(self.clf, x_train, y_train, params, 'accuracy')

    def fit_one(self):  # online fit
        pass

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def predict_proba(self, x_test):
        return pd.DataFrame(self.clf.predict_proba(x_test), columns=['-1', '1'], index=x_test.index)

    def evaluate(self, x_test, y_test):
        y_pred = self.clf.predict(x_test)

        accu = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        score = {'accu': accu, 'prec': prec, 'recall': recall, 'f1': f1}
        return score


