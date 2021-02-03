import numpy as np
import os.path as pth
import json
import models.ensgendel.samples_provider as SP

try:
    import cupy as cp
except:
    print("CUPY is not installed, setting cp to numpy")
    import numpy as cp



class Classifier(object):
    PFX_ARGS = "{}_args_{}.json"

    def __init__(self, gpu_on):
        self._gpu_on = gpu_on
        if gpu_on:
            self._xp = cp
        else:
            self._xp = np
        self.observer = {}

    def retype(self, array):
        if self._gpu_on:
            return self._xp.asarray(array)
        return array

    def untype(self, array):
        if self._gpu_on:
            return self._xp.asnumpy(array)
        return array

    def fit(self, batch_sample, batch_labels):
        pass

    def predict(self, samples):
        pass

    def evaluate(self, samples):
        """
        Returns scalary real value for each given sample.
        :param samples: samples to evaluate
        :return: evaluation over samples, numpy array (1, samples.shape[0])
        """
        pass

    def save_model(self, pathh, prefix=""):
        pass

    def save_args(self, pathh, prefix=""):
        pass

    def load_model(self, pathh, prefix=""):
        pass

    def _save_args(self, pathh, dictionary, prefix=""):
        name = self.PFX_ARGS.format(prefix, self.__class__.__name__)
        with open(pth.join(pathh, name), 'w') as file:
            file.write(json.dumps(dictionary))

    @classmethod
    def _create_from_args(cls, pathh, prefix="", units_clazz=None):
        name = cls.PFX_ARGS.format(prefix, cls.__name__)
        with open(pth.join(pathh, name), 'r') as file:
            dictionary = json.loads(file.read())
        return dictionary

    # def create_from_args(cls, pathh, prefix="")

    @classmethod
    def build_random(cls, **kwargs):
        raise NotImplementedError("not implemented for this class")

    @staticmethod
    def empty_sample_set(dim):
        return np.empty((0, dim), dtype=np.float32)


class ClassifierEnsemble(Classifier):
    PFX_UNIT = "{}_{}{}"

    def __init__(self, units, maximizing_units, gpu_on):
        super(ClassifierEnsemble, self).__init__(gpu_on)
        assert all([issubclass(unit.__class__, Classifier) for unit in units])
        self.units = units
        self.units_clazz = units[0].__class__.__name__
        self.units_num = len(units)
        self.samples_provider = None
        if maximizing_units:
            self.argbest = np.argmax
            self.best = np.max
            self.cmp = np.greater
        else:
            self.argbest = np.argmin
            self.best = np.min
            self.cmp = np.less

    def fit(self, batch_sample, batch_labels, **kwargs):
        pass

    def set_samples_provider(self, samples_provider):
        assert isinstance(samples_provider, SP.SamplesProvider)
        self.samples_provider = samples_provider

    def get_label_indexes(self, labels):
        indexes = np.arange(0, labels.shape[0])
        return [indexes[labels[:, 0] == i] for i in range(len(self.units))]

    def create_posneg_batch(self, positive_samples, negative_samples):
        posneg_batch = np.concatenate((
            positive_samples,
            negative_samples))
        posneg_labels = np.concatenate((
            np.ones((positive_samples.shape[0], 1), dtype=np.float32),
            np.zeros((negative_samples.shape[0], 1), dtype=np.float32)
        ))
        return posneg_batch, posneg_labels

    def predict(self, samples):
        return self.argbest(self.embed(samples), axis=1).reshape(-1, 1)

    def evaluate(self, samples):
        return self.best(self.embed(samples), axis=1).reshape(-1, 1)

    def embed(self, samples):
        return np.hstack([unit.evaluate(samples).reshape(-1, 1) for unit in self.units])

    def save_model(self, pathh, prefix=""):
        for i in range(len(self.units)):
            _prefix = self.PFX_UNIT.format(prefix, self.__class__.__name__, i)
            self.units[i].save_model(pathh, prefix=_prefix)

    def load_model(self, pathh, prefix=""):
        for i in range(len(self.units)):
            _prefix = self.PFX_UNIT.format(prefix, self.__class__.__name__, i)
            self.units[i].load_model(pathh, prefix=_prefix)

    def _save_args(self, pathh, dictionary, prefix=""):
        dictionary["units_num"] = self.units_num
        dictionary["units_clazz"] = self.units_clazz
        name = self.PFX_ARGS.format(prefix, self.__class__.__name__)
        with open(pth.join(pathh, name), 'w') as file:
            file.write(json.dumps(dictionary))

        for i in range(len(self.units)):
            _prefix = self.PFX_UNIT.format(prefix, self.__class__.__name__, i)
            self.units[i].save_args(pathh, prefix=_prefix)

    @classmethod
    def _create_from_args(cls, pathh, prefix="", units_clazz=None, overload_args=None):
        name = cls.PFX_ARGS.format(prefix, cls.__name__)

        with open(pth.join(pathh, name), 'r') as file:
            dictionary = json.loads(file.read())
        if units_clazz is None:
            units_clazz = eval(dictionary["units_clazz"])
        units = []
        for i in range(dictionary["units_num"]):
            _prefix = cls.PFX_UNIT.format(prefix, cls.__name__, i)
            units.append(units_clazz.create_from_args(pathh, prefix=_prefix, overload_args=overload_args))
            units[-1].load_model(pathh, prefix=_prefix)
        return units, dictionary




