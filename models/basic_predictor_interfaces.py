from incremental_evaluation import interfaces as I
from sklearn import linear_model
DEFAULT_CLASSES = (0, 1)


class LinearClassifierPredictor(I.Predictor):
    def __init__(self, classes=I.DEFAULT_CLASSES):
        super(LinearClassifierPredictor, self).__init__(classes=classes)
        self._clf = linear_model.SGDClassifier()
        self._fit = lambda X, y: self._clf.partial_fit(X, y, classes=self._classes)
        self._predict = self._clf.predict

    def __str__(self):
        return str(self.__class__.__name__)


class PerceptronClassifierPredictor(I.Predictor):
    def __init__(self, classes=I.DEFAULT_CLASSES):
        super(PerceptronClassifierPredictor, self).__init__(classes=classes)
        self._clf = linear_model.Perceptron()
        self._fit = lambda X, y: self._clf.partial_fit(X, y, classes=self._classes)
        self._predict = self._clf.predict

    def __str__(self):
        return str(self.__class__.__name__)