from sklearn.model_selection import GridSearchCV

class StandardTuner:

    def __init__(self):
        pass

    @staticmethod
    def tune_param(model, x_train, y_train, params, scoring):
        clf = GridSearchCV(model, params, cv=3, scoring=scoring)
        clf.fit(x_train, y_train)
        return clf
