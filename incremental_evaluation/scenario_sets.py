import incremental_evaluation.interfaces as I
import numpy as np
from incremental_evaluation import utils as IE
from chainer import datasets
import os


class FeatureMinimalScenarios(I.ScenarioSet):
    TRAIN_LABELS = "train_y.npy"
    TRAIN_SAMPLES = "train_x.npy"
    TEST_LABELS = "test_y.npy"
    TEST_SAMPLES = "test_x.npy"

    def __init__(self, dataset_path, digits_tripplet=(0, 1, 2), debug_set=False, scout_subset=None):
        super(FeatureMinimalScenarios, self).__init__()
        fir, sec, thr = digits_tripplet

        scenario_add = [{0: [fir]}, {1: [thr]}]
        scenario_exp = [{0: [fir]}, {1: [thr], 0: [sec]}]
        scenario_inc = [{0: [fir], 1: [sec]}, {1: [thr], 0: [sec]}]
        scenario_sep = [{0: [fir, sec]}, {1: [sec, thr]}]
        if not debug_set:
            self._scenarios = [
                scenario_add,
                # scenario_exp,
                # scenario_inc,
                # scenario_sep
            ]
        else:
            self._scenarios = [
                [{0: [fir], 1: [sec]}]
            ]

        self._training_samples, self._training_subclasses, self._testing_samples, self._testing_subclasses = (
            np.load(os.path.join(dataset_path, self.TRAIN_SAMPLES)),
            np.load(os.path.join(dataset_path, self.TRAIN_LABELS)),
            np.load(os.path.join(dataset_path, self.TEST_SAMPLES)),
            np.load(os.path.join(dataset_path, self.TEST_LABELS)),
        )
        self._training_subclasses = self._training_subclasses[:self._training_samples.shape[0]]
        if scout_subset is not None:
            ids = np.arange(self._training_samples.shape[0])
            np.random.shuffle(ids)
            self._training_samples = self._training_samples[ids[:scout_subset], :]
            self._training_subclasses = self._training_subclasses[ids[:scout_subset]]


class FeatureConvergentFiveScenarios(I.ScenarioSet):
    TRAIN_LABELS = "train_y.npy"
    TRAIN_SAMPLES = "train_x.npy"
    TEST_LABELS = "test_y.npy"
    TEST_SAMPLES = "test_x.npy"

    def __init__(self, dataset_path, scout_subset=None):
        super(FeatureConvergentFiveScenarios, self).__init__()

        ## Long tasks
        scenario_add5 = [{0: [0]}, {1: [1]}, {2: [2]}, {3: [3]}, {4: [4]}]
        scenario_exp5 = [{0: [0], 1: [1]}, {0: [2], 1: [3]}, {0: [4], 1: [5]}, {0: [6], 1: [7]}, {0: [8], 1: [9]}]
        scenario_sep5 = [{0: [1, 2, 3, 4, 5], 1: [0]}, {1: [1]}, {1: [2]}, {1: [3]}, {1: [4]}]

        self._scenarios = [
            scenario_add5,
            scenario_exp5,
            scenario_sep5,
        ]

        self._training_samples, self._training_subclasses, self._testing_samples, self._testing_subclasses = (
            np.load(os.path.join(dataset_path, self.TRAIN_SAMPLES)),
            np.load(os.path.join(dataset_path, self.TRAIN_LABELS)),
            np.load(os.path.join(dataset_path, self.TEST_SAMPLES)),
            np.load(os.path.join(dataset_path, self.TEST_LABELS)),
        )
        self._training_subclasses = self._training_subclasses[:self._training_samples.shape[0]]

        if scout_subset is not None:
            ids = np.arange(self._training_samples.shape[0])
            np.random.shuffle(ids)
            self._training_samples = self._training_samples[ids[:scout_subset], :]
            self._training_subclasses = self._training_subclasses[ids[:scout_subset]]

class MnistMinimalScenarios(I.ScenarioSet):
    def __init__(self, digits_tripplet=(0, 1, 2), debug_set=False, scout_subset=None):
        super(MnistMinimalScenarios, self).__init__()
        fir, sec, thr = digits_tripplet

        scenario_add = [{0: [fir]}, {1: [thr]}]
        scenario_exp = [{0: [fir]}, {1: [thr], 0: [sec]}]
        scenario_inc = [{0: [fir], 1: [sec]}, {1: [thr], 0: [sec]}]
        scenario_sep = [{0: [fir, sec]}, {1: [sec, thr]}]
        if not debug_set:
            self._scenarios = [
                scenario_add,
                scenario_exp,
                scenario_inc,
                scenario_sep
            ]
        else:
            self._scenarios = [
                [{0: [fir], 1: [sec]}]
            ]
        self._training_samples, train_sub, self._testing_samples, test_sub = mnist(list(digits_tripplet))
        self._training_subclasses = train_sub.reshape((-1,))
        self._testing_subclasses = test_sub.reshape((-1,))
        if scout_subset is not None:
            ids = np.arange(self._training_samples.shape[0])
            np.random.shuffle(ids)
            self._training_samples = self._training_samples[ids, :]
            self._training_subclasses = self._training_subclasses[ids]

            selected = []
            for i in digits_tripplet:
                selected.append((np.where(self._training_subclasses == i)[0])[:scout_subset])
            sels = np.concatenate(selected)

            self._training_samples = self._training_samples[sels, :]
            self._training_subclasses = self._training_subclasses[sels]


class MnistConvergentFiveScenarios(I.ScenarioSet):
    def __init__(self, scout_subset=None):
        super(MnistConvergentFiveScenarios, self).__init__()

        ## Long tasks
        scenario_add5 = [{0: [0]}, {1: [1]}, {2: [2]}, {3: [3]}, {4: [4]}]
        scenario_exp5 = [{0: [0], 1: [1]}, {0: [2], 1: [3]}, {0: [4], 1: [5]}, {0: [6], 1: [7]}, {0: [8], 1: [9]}]
        scenario_sep5 = [{0: [1, 2, 3, 4, 5], 1: [0]}, {1: [1]}, {1: [2]}, {1: [3]}, {1: [4]}]

        self._scenarios = [
            scenario_add5,
            scenario_exp5,
            scenario_sep5,
        ]

        self._training_samples, train_sub, self._testing_samples, test_sub = mnist(list(range(10)))
        self._training_subclasses = train_sub.reshape((-1,))
        self._testing_subclasses = test_sub.reshape((-1,))
        if scout_subset is not None:
            ids = np.arange(self._training_samples.shape[0])
            np.random.shuffle(ids)
            self._training_samples = self._training_samples[ids, :]
            self._training_subclasses = self._training_subclasses[ids]

            selected = []
            for i in range(10):
                selected.append((np.where(self._training_subclasses == i)[0])[:scout_subset])
            sels = np.concatenate(selected)

            self._training_samples = self._training_samples[sels, :]
            self._training_subclasses = self._training_subclasses[sels]


class Gauss3DMinimalScenarios(I.ScenarioSet):
    VARIANCE_CRISP = 0.01
    R_MEAN = [1, -1, -1]
    G_MEAN = [-1, 1, -1]
    B_MEAN = [-1, -1, 1]
    COV_CRISP = [[VARIANCE_CRISP, 0, 0], [0, VARIANCE_CRISP, 0], [0, 0, VARIANCE_CRISP]]

    def __init__(self, means=(R_MEAN, G_MEAN, B_MEAN), cov=COV_CRISP, train_size=2000, test_size=500):
        super(Gauss3DMinimalScenarios, self).__init__()
        size = train_size + test_size
        samples = np.concatenate([np.random.multivariate_normal(means[i], cov, size) for i in range(len(means))])
        subclasses = np.concatenate([np.ones((size, ), dtype=np.int) * i for i in range(len(means))])
        ids = np.arange(subclasses.shape[0])
        np.random.shuffle(ids)
        train_total_size = train_size * len(means)
        self._training_samples = samples[ids[:train_total_size], :]
        self._training_subclasses = subclasses[ids[:train_total_size]]
        self._testing_samples = samples[ids[train_total_size:], :]
        self._testing_subclasses = subclasses[ids[train_total_size:]]
        self._scenarios = [
            IE.SCENARIO_ADDITION,
            IE.SCENARIO_EXPANSION,
            IE.SCENARIO_INCLUSION,
            IE.SCENARIO_SEPARATION
        ]

def mnist(selected_labels_task):
    train, test = datasets.get_mnist()
    trn_set, trn_labels = train._datasets
    tst_set, tst_labels = test._datasets

    task_trn_condition = get_condition(trn_labels, selected_labels_task)
    task_tst_condition = get_condition(tst_labels, selected_labels_task)
    return (
        trn_set[task_trn_condition].astype(np.float32),
        trn_labels[task_trn_condition].reshape((-1, 1)).astype(np.float32),
        tst_set[task_tst_condition].astype(np.float32),
        tst_labels[task_tst_condition].reshape((-1, 1)).astype(np.float32),
    )


def get_condition(labels, selected_labels):
    _condition = [labels == i for i in selected_labels]
    _condition = np.asarray(_condition, dtype=np.bool_)
    _condition = np.sum(_condition, axis=0, dtype=np.bool_)
    return _condition
