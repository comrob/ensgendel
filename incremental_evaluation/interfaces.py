import numpy as np
DEFAULT_CLASSES = (0, 1)
"""
These interfaces should be used if you want to expand this framework with new predictors and scenario sets.
"""

class Predictor(object):
    def __init__(self, classes=DEFAULT_CLASSES):
        """
        The interface for classifier learning algorithms. The only constructor input is optional classes. Nothing
        else is expected. The parametrization should be done within this class.
        @param classes: The only input
        """
        self._predict = lambda X: np.empty((0, ))
        self._fit = lambda X, y: None
        self._classes = list(classes)

    def predict(self, X):
        """
        Transforms features X into labels y.
        @param X: np.ndarray of shape (number_of_instances, feature_dimension)
        @return: feature labels np.ndarray of shape (number_of_instances, ) and type np.int
        """
        return self._predict(X)

    def fit(self, X, y):
        """
        Trains on given training set. This is a modifier type function, it should modify the model stored in this class.
        @param X: features, np.ndarray of shape (number_of_instances, feature_dimension)
        @param y: feature labels np.ndarray of shape (number_of_instances, ) and type np.int
        @return: optional, can be None.
        """
        return self._fit(X, y)


class ScenarioSet(object):
    def __init__(self):
        """
        Class producing set of scenarios constructed from subclasses.
        Each scenario is defined as a list of tasks, and each task is defined as a label-subclasses assignments:

        [{hard_terrain: [swamp], easy_terrain: [road, snow]}, {hard_terrain: [snow]}]
        Here we have two tasks. In the first one the classifier gets images of swamp labeled as hard_terrain and
        the road and snow images are labeled as easy_terrain. In the second task the classifier trains on a new dataset
        where the snow images have hard_terrain label (label of snow changed). The testing task would be instances of
        swamp and snow labeled as hard_terrain and road images labeled as easy_terrain.

        In this example the swamp, road, and snow are the subclasses, where the images are samples and
        terrain types identifiers are subclasses. These subclasses are explicitly known only to scenario
        construction methods. The classifier then gets only the images and labels.
        In the case of the example there are two labels: hard_terrain, easy_terrain.

        In practice, we index the subclasses and labels with integers. So the example scenario would be encoded as
        [{0: [0], 1: [1, 2]}, {0: [2]}]
        """
        # Overwrite this constructor and the private attributes.
        self._scenarios = []  # list of scenarios [[{label: [subclass, ...], ...}, ...], ...]
        self._testing_subclasses = np.empty((0, ))  # testing subclasses shape: (num_of_test_samples, )
        self._testing_samples = np.empty((0, 0))  # testing samples, shape: (num_of_test_samples, feature_dimension)
        self._training_subclasses = np.empty((0, ))  # testing subclasses shape: (num_of_train_samples, )
        self._training_samples = np.empty((0, 0))  # testing samples, shape: (num_of_train_samples, feature_dimension)

    def get_training_set(self):
        #  Do not overwrite this
        return self._training_samples, self._training_subclasses

    def get_test_set(self):
        #  Do not overwrite this
        return self._testing_samples, self._testing_subclasses

    def get_scenarios(self):
        #  Do not overwrite this
        return self._scenarios





