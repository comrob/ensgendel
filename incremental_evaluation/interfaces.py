import numpy as np
DEFAULT_CLASSES = (0, 1)


class Predictor(object):
    def __init__(self, classes=DEFAULT_CLASSES):
        self._predict = lambda X: np.empty((0, ))
        self._fit = lambda X, y: None
        self._classes = list(classes)

    def predict(self, X):
        return self._predict(X)

    def fit(self, X, y):
        return self._fit(X, y)


class ScenarioSet(object):
    def __init__(self):
        self._scenarios = []
        self._testing_subclasses = np.empty((0, ))
        self._testing_samples = np.empty((0, 0))
        self._training_subclasses = np.empty((0, ))
        self._training_samples = np.empty((0, 0))

    def get_training_set(self):
        return self._training_samples, self._training_subclasses

    def get_test_set(self):
        return self._testing_samples, self._testing_subclasses

    def get_scenarios(self):
        return self._scenarios





