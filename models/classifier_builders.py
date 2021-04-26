from models.ensgendel.ensgendel_classifier_ensemble import AddDelEnsembleClassifier
from models.ensgendel.variational_autoencoder import VariationalAutoencoder, CnnVariationalAutoencoder

"""
This file contains all tested classifiers.
"""
EMPTY_ARGS = dict()


class ClassifierMetaContainer(object):
    def __init__(self, classifier_clazz, classifier_builder, name):
        """
        Classifier container replaces tuple in samples_provider.
        :param classifier:
        :param name:
        """
        self.clazz = classifier_clazz
        self.builder = classifier_builder
        self.name = name


def getelse(args, key, default):
    return args[key] if key in args else default


def cnn_adevae(args=EMPTY_ARGS):
    units_number = getelse(args, "units_number", 10)
    feat_size = getelse(args, "feat_size", 28)
    hidden_size = getelse(args, "hidden_size", 500)
    latent_size = getelse(args, "latent_size", 8)
    gpu_on = getelse(args, "gpu_on", False)
    mini_batch = getelse(args, "mini_batch", 124)
    threshold = getelse(args, "threshold", 0.1)
    subtract_epsilon = getelse(args, "subtract_epsilon", 4)
    max_updates = getelse(args, "max_updates", 100)
    min_updates = getelse(args, "min_updates", 10)
    learning_rate = getelse(args, "learning_rate", 0.001)
    max_gen_samples = getelse(args, "max_gen_samples", 200)
    channels = getelse(args, "channels", 3)

    def build():

        return AddDelEnsembleClassifier(
            [
                CnnVariationalAutoencoder(feat_size=feat_size, hidden_size=hidden_size, latent_size=latent_size,
                                       learning_rate=learning_rate, gpu_on=gpu_on, mini_batch=mini_batch,
                                       threshold=threshold, channels=channels)
                for _ in range(units_number)
            ],
            threshold=threshold, max_updates=max_updates, min_updates=min_updates, subtract_epsilon=subtract_epsilon,
            maximizing_units=False, gpu_on=gpu_on, uncover_off=False, max_gen_samples=max_gen_samples)

    return ClassifierMetaContainer(AddDelEnsembleClassifier, build, "cnn_adevae")

def adevae(args=EMPTY_ARGS):
    units_number = getelse(args, "units_number", 10)
    feat_size = getelse(args, "feat_size", 784)
    hidden_size = getelse(args, "hidden_size", 800)
    latent_size = getelse(args, "latent_size", 500)
    gpu_on = getelse(args, "gpu_on", False)
    mini_batch = getelse(args, "mini_batch", 124)
    threshold = getelse(args, "threshold", 0.5)
    subtract_epsilon = getelse(args, "subtract_epsilon", 0.1)
    max_updates = getelse(args, "max_updates", 50)
    min_updates = getelse(args, "min_updates", 10)
    learning_rate = getelse(args, "learning_rate", 0.001)
    max_gen_samples = getelse(args, "max_gen_samples", 200)

    def build():

        return AddDelEnsembleClassifier(
            [
                VariationalAutoencoder(feat_size=feat_size, hidden_size=hidden_size, latent_size=latent_size,
                                       learning_rate=learning_rate, gpu_on=gpu_on, mini_batch=mini_batch,
                                       threshold=threshold)
                for _ in range(units_number)
            ],
            threshold=threshold, max_updates=max_updates, min_updates=min_updates, subtract_epsilon=subtract_epsilon,
            maximizing_units=False, gpu_on=gpu_on, uncover_off=False, max_gen_samples=max_gen_samples)

    return ClassifierMetaContainer(AddDelEnsembleClassifier, build, "adevae")


def aevae(args=EMPTY_ARGS):
    units_number = getelse(args, "units_number", 10)
    feat_size = getelse(args, "feat_size", 784)
    hidden_size = getelse(args, "hidden_size", 800)
    latent_size = getelse(args, "latent_size", 500)
    gpu_on = getelse(args, "gpu_on", False)
    mini_batch = getelse(args, "mini_batch", 124)
    threshold = getelse(args, "threshold", 0.5)
    subtract_epsilon = getelse(args, "subtract_epsilon", 0.1)
    max_updates = getelse(args, "max_updates", 50)
    min_updates = getelse(args, "min_updates", 10)
    learning_rate = getelse(args, "learning_rate", 0.001)
    max_gen_samples = getelse(args, "max_gen_samples", 200)
    def build():

        return AddDelEnsembleClassifier(
            [
                VariationalAutoencoder(feat_size=feat_size, hidden_size=hidden_size, latent_size=latent_size,
                                       learning_rate=learning_rate, gpu_on=gpu_on, mini_batch=mini_batch,
                                       threshold=threshold)
                for _ in range(units_number)
            ],
            threshold=threshold, max_updates=max_updates, min_updates=min_updates, subtract_epsilon=subtract_epsilon,
            maximizing_units=False, gpu_on=gpu_on, uncover_off=True, max_gen_samples=max_gen_samples)

    return ClassifierMetaContainer(AddDelEnsembleClassifier, build, "aevae")

def evae(args=EMPTY_ARGS):
    units_number = getelse(args, "units_number", 10)
    feat_size = getelse(args, "feat_size", 784)
    hidden_size = getelse(args, "hidden_size", 800)
    latent_size = getelse(args, "latent_size", 500)
    gpu_on = getelse(args, "gpu_on", False)
    mini_batch = getelse(args, "mini_batch", 124)
    threshold = getelse(args, "threshold", 0.5)
    subtract_epsilon = getelse(args, "subtract_epsilon", 0.1)
    max_updates = getelse(args, "max_updates", 50)
    min_updates = getelse(args, "min_updates", 10)
    learning_rate = getelse(args, "learning_rate", 0.001)
    def build():

        return AddDelEnsembleClassifier(
            [
                VariationalAutoencoder(feat_size=feat_size, hidden_size=hidden_size, latent_size=latent_size,
                                       learning_rate=learning_rate, gpu_on=gpu_on, mini_batch=mini_batch,
                                       threshold=threshold)
                for _ in range(units_number)
            ],
            threshold=threshold, max_updates=max_updates, min_updates=min_updates, subtract_epsilon=subtract_epsilon,
            maximizing_units=False, gpu_on=gpu_on, uncover_off=True, max_gen_samples=0)

    return ClassifierMetaContainer(AddDelEnsembleClassifier, build, "evae")