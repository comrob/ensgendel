import incremental_evaluation.interfaces as I
from models.classifier_builders import adevae
import models.ensgendel.samples_provider as SP
import numpy as np


class Ensgendel(I.Predictor):
    def __init__(self, classes, max_epoch=2, gpu_on=False):
        super().__init__(classes)
        self._max_epoch = max_epoch
        self._class_num = len(classes)
        # common params
        self._predictor_args = {"units_number": self._class_num}
        self._predictor_args["gpu_on"] = gpu_on
        self._predictor_args["mini_batch"] = 128
        self._predictor_args["max_epoch"] = max_epoch
        self._predictor_args["min_updates"] = 0
        self._predictor_args["learning_rate"] = 0.001

        self._is_first_fit = True

    def _before_first_fit(self, X, y):
        self._is_first_fit = False
        #domain specific parameters
        self._predictor_args["feat_size"] = X.shape[1]
        if self._predictor_args["feat_size"] > 3:
            # tuned for MNIST input
            max_updates = 10
            hidden_size = 500
            latent_size = 8
            threshold = 0.1
            subtract_epsilon = 4
        else:
            # tuned for 3D gauss clusters
            max_updates = 100
            hidden_size = 30
            latent_size = 2
            threshold = 0.2
            subtract_epsilon = 0.3

        self._predictor_args["hidden_size"] = hidden_size
        self._predictor_args["latent_size"] = latent_size
        self._predictor_args["threshold"] = threshold
        self._predictor_args["max_updates"] = max_updates
        self._predictor_args["subtract_epsilon"] = subtract_epsilon

        self._predictor = adevae(self._predictor_args).builder()
        samples_provider = SP.CompactGanSampleProvider(self._predictor.units)
        self._predictor.set_samples_provider(samples_provider)

    def fit(self, X, y):
        if self._is_first_fit:
            self._before_first_fit(X, y)
        ####
        sampling_on = True
        fragment_set_size = 256
        batch_size_max = None
        max_epoch = self._max_epoch
        train_samples = X.astype(dtype=np.float32)
        train_labels = y.reshape((-1, 1))
        ####

        if fragment_set_size is None:
            fragment_set_size = train_samples.shape[0]

        # configuration
        if batch_size_max is None:
            batch_size_max = train_samples.shape[0]
            iterative_epochs = 1
        else:
            iterative_epochs = (train_samples.shape[0] // batch_size_max) + 1

        for iterative_epoch in range(iterative_epochs):
            print("Iterative epoch {}/{}".format(iterative_epoch + 1, iterative_epochs))
            iter_batch_start = iterative_epoch * batch_size_max
            if iterative_epoch != 0:
                batch_size = min(batch_size_max, (train_samples.shape[0] - iter_batch_start) % batch_size_max)
            else:
                batch_size = train_samples.shape[0]
            for epoch in range(max_epoch):
                kwargs = {}
                kwargs["sampling_on"] = sampling_on
                if epoch == max_epoch - 1:
                    kwargs["last_epoch"] = True
                print("batch_size:{}, iter_batch_start:{}".format(batch_size, iter_batch_start))
                indexes = np.random.permutation(batch_size - 1) + iter_batch_start
                train_samples_mini = train_samples[indexes[:fragment_set_size], :]
                train_labels_mini = train_labels[indexes[:fragment_set_size], :]
                self._predictor.fit(train_samples_mini, train_labels_mini, kwargs)

    def predict(self, X):
        res = self._predictor.predict(X.astype(dtype=np.float32))
        return res.reshape((-1,))
