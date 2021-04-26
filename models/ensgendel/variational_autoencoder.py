from models.ensgendel.classifier import *
from chainer import optimizers
from chainer import Variable, Chain
from chainer import functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
from chainer import no_backprop_mode

import numpy as np
xp = np


class VariationalAutoencoder(Classifier):
    NAME = "VariationalAutoencoder"

    class Model(Chain):

        def __init__(self, feat_size, hidden_size, latent_size, xp=np, **links):
            super(VariationalAutoencoder.Model, self).__init__(**links)
            self.feat_size = feat_size
            with self.init_scope():
                # encoder
                self.e1 = L.Linear(feat_size, hidden_size//2)
                self.e12 = L.Linear(hidden_size//2, hidden_size)
                self.e2_mu = L.Linear(hidden_size, latent_size)
                self.e2_ln_var = L.Linear(hidden_size, latent_size)
                # decoder
                self.d1 = L.Linear(latent_size, hidden_size//2)
                self.d12 = L.Linear(hidden_size//2, hidden_size)
                self.d2 = L.Linear(hidden_size, feat_size + 1)  # dimension extension

        def __call__(self, x):
            return self.decode(self.encode(x)[0])

        def encode(self, x):
            h1 = F.relu(self.e1(x))
            h12 = F.relu(self.e12(h1))
            mu = self.e2_mu(h12)
            ln_var = self.e2_ln_var(h12)
            return mu, ln_var

        def decode(self, z):
            h1 = F.relu(self.d1(z))
            h12 = F.relu(self.d12(h1))
            h_out = self.d2(h12)
            h_feature_space, h_extension = h_out[:, :self.feat_size],  h_out[:, self.feat_size:]
            return h_feature_space, h_extension

        def save_model(self, pathh, prefix=""):
            serializers.npz.save_npz(pth.join(pathh, "{}_{}".format(prefix, self.__class__.__name__)), self)

        def load_model(self, pathh, prefix=""):
            serializers.load_npz(pth.join(pathh, "{}_{}".format(prefix, self.__class__.__name__)), self)

    def __init__(self, feat_size, hidden_size, latent_size, learning_rate=0.01, gpu_on=False, mini_batch=256,
                 threshold=0.5, extended_dimension=False, threshold_extended=0.1):
        super(VariationalAutoencoder, self).__init__(gpu_on)
        self._args = feat_size, hidden_size, latent_size, learning_rate, gpu_on, mini_batch, threshold, extended_dimension, threshold_extended
        self._args_names = "feat_size, hidden_size, latent_size, learning_rate, gpu_on, mini_batch, threshold, extended_dimension, threshold_extended"

        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.observer = {"loss": 0, "rec_loss": 0, "reg_loss": 0, "cost": 0, "cost_decoder": 0, "cost_encoder": 0}
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch
        self.optimiser = None
        self.model = self.Model(feat_size, hidden_size, latent_size, xp=self._xp)
        self.threshold = threshold
        self.threshold_extended = threshold_extended
        self.latent_samples_generator = self.normal_sampling
        # LOSS params
        self.neg_loss_weight = 0
        self.pos_loss_weight = 1
        self.extension_loss_weight = 1
        self.rbf_beta = 1
        self.alpha_1 = 1
        self.alpha_2 = 1

        self.extension_loss = self._extension_squared_loss
        # CLASSIFIER SETUP
        self.gpu_on = gpu_on
        if gpu_on:
            self.model.to_gpu()

    def _get_optimiser(self):
        if self.optimiser is None:
            self.optimiser = optimizers.Adam(alpha=self.learning_rate)
            self.optimiser.setup(self.model)
            # self.optimiser.lr=0
            # self.optimiser.add_hook(WeightDecay(0.0005))
        return self.optimiser

    def remodel(self):
        self.model.cleargrads()
        #self.model = self.Model(self.feat_size, self.hidden_size, self.latent_size, xp=self._xp)
        #if self.gpu_on:
        #    self.model.to_gpu()

    def _extension_bernoulli_loss(self, labels, extension):
        return F.bernoulli_nll(labels, extension)

    def _extension_squared_loss(self, labels, extension):
        _labels = labels.reshape((-1, 1))
        return F.sum(F.squared_difference(F.sigmoid(extension), _labels))

    def _back_propagation(self, real_samples, labels, resampling_n=20):
        """
        One backpropagation step.
        :param real_samples: real_samples taken from input space
        :param labels: labels real_samples 1 - in samples, 0 - out samples
        :return:
        """
        batchsize = real_samples.shape[0] + 0.00001
        _real_samples = Variable(real_samples)
        _pos_num = self._xp.sum(labels)
        _labels = labels.reshape((-1,))
        mu, ln_var = self.model.encode(_real_samples)
        # negative_feature_error = 0
        positive_feature_error = 0
        extension_error = 0
        for _ in range(resampling_n):
            z = F.gaussian(mu, ln_var)
            feature, extension = self.model.decode(z)
            # Reconstruction cost is evaluated only for positive samples (label = 1)

            positive_feature_error += F.sum(
                _labels * F.sum(
                    F.squared_difference(_real_samples, feature), axis=1
                )
            )

        # EXTENSION ERROR
        extension_error += self.extension_loss(_labels, self.model.decode(mu)[1])

        rec_loss = self.pos_loss_weight * positive_feature_error
        rec_loss += self.extension_loss_weight * extension_error
        rec_loss /= resampling_n
        reg_loss = F.sum(_labels * F.sum(gaussian_kl_divergence(mu, F.clip(ln_var, -50., 50.), reduce='no'), axis=1))
        print("                                                                                 "
              " ext: {}, pos: {}, reg:{}".format(extension_error.data, positive_feature_error.data, reg_loss.data))
        assert not np.isnan(reg_loss.data), "There is a nan"
        # Regularization cost for negative samples (label = 0) penalize the closeness of the unit gauss.

        loss = self.alpha_1 * rec_loss + self.alpha_2 * reg_loss
        loss /= batchsize

        # update
        loss.cleargrad()
        self._get_optimiser().target.cleargrads()
        loss.backward()
        self._get_optimiser().update()
        # OBSERVATIONS
        self.observer["loss"] += loss.data
        self.observer["rec_loss"] += rec_loss.data / batchsize
        self.observer["reg_loss"] += reg_loss.data / batchsize

    def fit(self, real_samples, labels):
        assert real_samples.ndim == 2
        assert labels.ndim == 2
        assert labels.shape[0] == real_samples.shape[0]
        assert real_samples.shape[1] == self.feat_size, "Feature dimension should be {} but it is {}.".format(
            self.feat_size, real_samples.shape[0])
        assert labels.shape[1] == 1

        self.observer["loss"] = 0
        self.observer["rec_loss"] = 0
        self.observer["reg_loss"] = 0

        # GPU RETYPING
        real_samples = self.retype(real_samples)
        labels = self.retype(labels)
        # PROCESSING DATA
        if self.mini_batch is None:
            self._back_propagation(real_samples, labels)
        else:
            indexes = self._xp.random.permutation(real_samples.shape[0])
            iternum = indexes.shape[0] // self.mini_batch
            if indexes.shape[0] % self.mini_batch > 0:
                iternum += 1
            for i in range(iternum):
                minibatch_indexes = indexes[self.mini_batch * i:self.mini_batch * (i + 1)]
                minibatch_real_samples = real_samples[minibatch_indexes, :]
                minibatch_labels = labels[minibatch_indexes, :]
                self._back_propagation(minibatch_real_samples, minibatch_labels)
            self.observer["loss"] /= float(iternum)
            self.observer["rec_loss"] /= float(iternum)
            self.observer["reg_loss"] /= float(iternum)
            # observations with different names for COMPATIBILIIY
            self.observer["cost"] = self.observer["loss"]
            self.observer["cost_decoder"] = self.observer["rec_loss"]
            self.observer["cost_encoder"] = self.observer["reg_loss"]

    def get_imitations(self, n):
        return self.untype(
            self.model.decode(Variable(self.retype(self.latent_samples_generator(n, self.latent_size))))[0].data)

    def predict(self, samples, threshold=None, **argv):
        if threshold is None:
            threshold = self.threshold
        evaluations = self.evaluate(samples)
        return (evaluations < threshold).reshape((-1, 1))

    def evaluate(self, samples):
        """
        Returns scalary real value for each given sample.
        :param samples: samples to evaluate
        :return: evaluation over samples, numpy array (1, samples.shape[0])
        """
        if len(samples) == 0:
            return []
        samples = self.retype(samples)
        results = []
        indexes = self._xp.arange(0, samples.shape[0])
        iternum = indexes.shape[0] // self.mini_batch
        if indexes.shape[0] % self.mini_batch > 0:
            iternum += 1
        with no_backprop_mode():
            for i in range(iternum):
                minibatch_indexes = indexes[self.mini_batch * i:self.mini_batch * (i + 1)]
                minisamples = samples[minibatch_indexes, :]
                mu, ln_var = self.model.encode(minisamples)
                feature, extension = self.model.decode(mu)
                loss = 1 - F.sigmoid(extension)[:, 0]  # distance from one
                loss += F.sum(
                    F.squared_difference(minisamples, feature), axis=1) / self.feat_size
                results.append(loss.data)
        return self.untype(self._xp.concatenate(results))

    def save_model(self, pathh, prefix=""):
        self.model.save_model(pathh, prefix=prefix + "_{}".format(self.__class__.__name__))

    def save_args(self, pathh, prefix=""):
        arguments = [ar.strip() for ar in self._args_names.split(',')]
        dictionary = dict(zip(arguments, self._args))
        self._save_args(pathh, dictionary, prefix=prefix)

    def load_model(self, pathh, prefix=""):
        self.model.load_model(pathh, prefix=prefix + "_{}".format(self.__class__.__name__))

    @classmethod
    def create_from_args(cls, pathh, prefix="", units_clazz=None, overload_args=None):
        dictionary = cls._create_from_args(pathh, prefix=prefix, units_clazz=units_clazz)
        if overload_args is not None:
            for koa in overload_args:
                dictionary[koa] = overload_args[koa]
        return cls(feat_size=dictionary["feat_size"], hidden_size=dictionary["hidden_size"],
                   latent_size=dictionary["latent_size"], mini_batch=dictionary["mini_batch"],
                   learning_rate=dictionary["learning_rate"], gpu_on=dictionary["gpu_on"],
                   threshold=dictionary["threshold"],extended_dimension=dictionary["extended_dimension"],
                   threshold_extended=dictionary["threshold_extended"]
                   )

    @classmethod
    def build_random(cls, **kwargs):
        raise NotImplementedError("not implemented for this class")

    @staticmethod
    def empty_sample_set(dim):
        return np.empty((0, dim), dtype=np.float32)

    def __str__(self):
        arguments = [ar.strip() for ar in self._args_names.split(',')]
        dictionary = dict(zip(arguments, self._args))
        return "{} with arguments {}".format(self.__class__.__name__, dictionary)

    @staticmethod
    def normal_sampling(n, latent_space):
        return np.random.standard_normal((n, latent_space)).astype(dtype=np.float32)


class CnnVariationalAutoencoder(Classifier):
    NAME = "CnnVariationalAutoencoder"

    class Model(Chain):

        def __init__(self, feat_size, hidden_size, latent_size, channels, xp=np, **links):
            super(CnnVariationalAutoencoder.Model, self).__init__(**links)
            self.feat_size = feat_size
            self.channels = channels
            self.hidden_size = hidden_size
            cnn_depth = 4
            self.bottom_feat_size = feat_size // pow(2, cnn_depth)
            with self.init_scope():
                # encoder
                self.cnn_in = L.Convolution2D(channels, hidden_size // 8, ksize=4, stride=2, pad=1)
                self.cnn_1 = L.Convolution2D(hidden_size // 8, hidden_size // 4, ksize=4, stride=2, pad=1)
                self.cnn_2 = L.Convolution2D(hidden_size // 4, hidden_size // 2, ksize=4, stride=2, pad=1)
                self.cnn_3 = L.Convolution2D(hidden_size // 2, hidden_size, ksize=2, stride=2, pad=0)
                self.bn_1 = L.BatchNormalization(hidden_size // 4, use_gamma=False)
                self.bn_2 = L.BatchNormalization(hidden_size // 2, use_gamma=False)
                self.bn_3 = L.BatchNormalization(hidden_size, use_gamma=False)

                self.e1 = L.Linear(self.bottom_feat_size * self.bottom_feat_size * hidden_size, hidden_size)
                self.e2_mu = L.Linear(hidden_size, latent_size)
                self.e2_ln_var = L.Linear(hidden_size, latent_size)
                # decoder
                self.d1 = L.Linear(latent_size, hidden_size)
                self.d2 = L.Linear(hidden_size, self.bottom_feat_size * self.bottom_feat_size * hidden_size)
                self.dcnn_3 = L.Deconvolution2D(hidden_size, hidden_size//2, ksize=2, stride=2, pad=0)
                self.dcnn_2 = L.Deconvolution2D(hidden_size//2, hidden_size//4, ksize=4, stride=2, pad=1)
                self.dcnn_1 = L.Deconvolution2D(hidden_size//4, hidden_size//8, ksize=4, stride=2, pad=1)
                self.dcnn_out = L.Deconvolution2D(hidden_size//8, channels, ksize=4, stride=2, pad=1)
                self.dbn_3 = L.BatchNormalization(hidden_size//2, use_gamma=False)
                self.dbn_2 = L.BatchNormalization(hidden_size // 4, use_gamma=False)
                self.dbn_1 = L.BatchNormalization(hidden_size // 8, use_gamma=False)

                # extension
                self.ex_out = L.Linear(self.bottom_feat_size * self.bottom_feat_size * hidden_size, 1)


        def __call__(self, x):
            return self.decode(self.encode(x)[0])

        def encode(self, x):
            h1 = F.relu(self.cnn_in(x))
            h2 = F.relu(self.bn_1(self.cnn_1(h1)))
            h3 = F.relu(self.bn_2(self.cnn_2(h2)))
            h4 = F.relu(self.bn_3(self.cnn_3(h3)))

            h_e1 = F.relu(self.e1(h4))
            mu = self.e2_mu(h_e1)
            ln_var = self.e2_ln_var(h_e1)
            return mu, ln_var

        def decode(self, z):
            h_d1 = F.relu(self.d1(z))
            h_d2 = F.relu(self.d2(h_d1))
            h3 = F.relu(self.dbn_3(self.dcnn_3(
                F.reshape(h_d2, (len(z), self.hidden_size, self.bottom_feat_size, self.bottom_feat_size))
            )))
            h2 = F.relu(self.dbn_2(self.dcnn_2(h3)))
            h1 = F.relu(self.dbn_1(self.dcnn_1(h2)))
            h_out = F.sigmoid(self.dcnn_out(h1))
            # extension
            ex = self.ex_out(h_d2)

            return h_out, ex

        def save_model(self, pathh, prefix=""):
            serializers.npz.save_npz(pth.join(pathh, "{}_{}".format(prefix, self.__class__.__name__)), self)

        def load_model(self, pathh, prefix=""):
            serializers.load_npz(pth.join(pathh, "{}_{}".format(prefix, self.__class__.__name__)), self)

    def __init__(self, feat_size, hidden_size, latent_size, learning_rate=0.01, gpu_on=False, mini_batch=256,
                 threshold=0.5, extended_dimension=False, threshold_extended=0.1, channels=3):
        super(CnnVariationalAutoencoder, self).__init__(gpu_on)
        self._args = feat_size, hidden_size, latent_size, learning_rate, gpu_on, mini_batch, threshold, extended_dimension, threshold_extended, channels
        self._args_names = "feat_size, hidden_size, latent_size, learning_rate, gpu_on, mini_batch, threshold, extended_dimension, threshold_extended, channels"
        self.channels = channels
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.observer = {"loss": 0, "rec_loss": 0, "reg_loss": 0, "cost": 0, "cost_decoder": 0, "cost_encoder": 0}
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch
        self.optimiser = None
        self.model = self.Model(feat_size, hidden_size, latent_size, xp=self._xp, channels=channels)
        self.threshold = threshold
        self.threshold_extended = threshold_extended
        self.latent_samples_generator = self.normal_sampling
        self.feature_volume = self.feat_size * self.feat_size * self.channels
        # LOSS params
        self.neg_loss_weight = 0
        self.pos_loss_weight = 1
        self.extension_loss_weight = 1
        self.rbf_beta = 1
        self.alpha_1 = 1
        self.alpha_2 = 1

        self.extension_loss = self._extension_squared_loss
        # CLASSIFIER SETUP
        self.gpu_on = gpu_on
        if gpu_on:
            self.model.to_gpu()

    def _get_optimiser(self):
        if self.optimiser is None:
            self.optimiser = optimizers.Adam(alpha=self.learning_rate)
            self.optimiser.setup(self.model)
            # self.optimiser.lr=0
            # self.optimiser.add_hook(WeightDecay(0.0005))
        return self.optimiser

    def remodel(self):
        self.model.cleargrads()
        #self.model = self.Model(self.feat_size, self.hidden_size, self.latent_size, xp=self._xp)
        #if self.gpu_on:
        #    self.model.to_gpu()

    def _extension_bernoulli_loss(self, labels, extension):
        return F.bernoulli_nll(labels, extension)

    def _extension_squared_loss(self, labels, extension):
        _labels = labels.reshape((-1, 1))
        return F.sum(F.squared_difference(F.sigmoid(extension), _labels))

    def _back_propagation(self, real_samples, labels, resampling_n=20):
        """
        One backpropagation step.
        :param real_samples: real_samples taken from input space
        :param labels: labels real_samples 1 - in samples, 0 - out samples
        :return:
        """
        batchsize = real_samples.shape[0] + 0.00001
        _real_samples = Variable(real_samples)
        _pos_num = self._xp.sum(labels)
        _labels = labels.reshape((-1,))
        mu, ln_var = self.model.encode(_real_samples)
        # negative_feature_error = 0
        positive_feature_error = 0
        extension_error = 0
        for _ in range(resampling_n):
            z = F.gaussian(mu, ln_var)
            feature, extension = self.model.decode(z)
            # Reconstruction cost is evaluated only for positive samples (label = 1)

            positive_feature_error += F.sum(
                _labels * F.reshape(F.sum_to(F.squared_difference(_real_samples, feature), (len(_real_samples), 1, 1, 1)), (-1, ))
            )

        # EXTENSION ERROR
        extension_error += self.extension_loss(_labels, self.model.decode(mu)[1])

        rec_loss = self.pos_loss_weight * positive_feature_error
        rec_loss += self.extension_loss_weight * extension_error
        rec_loss /= resampling_n
        reg_loss = F.sum(_labels * F.sum(gaussian_kl_divergence(mu, F.clip(ln_var, -50., 50.), reduce='no'), axis=1))
        print("                                                                                 "
              " ext: {}, pos: {}, reg:{}".format(extension_error.data, positive_feature_error.data, reg_loss.data))
        assert not np.isnan(reg_loss.data), "There is a nan"
        # Regularization cost for negative samples (label = 0) penalize the closeness of the unit gauss.

        loss = self.alpha_1 * rec_loss + self.alpha_2 * reg_loss
        loss /= batchsize

        # update
        loss.cleargrad()
        self._get_optimiser().target.cleargrads()
        loss.backward()
        self._get_optimiser().update()
        # OBSERVATIONS
        self.observer["loss"] += loss.data
        self.observer["rec_loss"] += rec_loss.data / batchsize
        self.observer["reg_loss"] += reg_loss.data / batchsize

    def _vec_to_image(self, X):
        return X.reshape(X.shape[0], self.channels, self.feat_size, self.feat_size)

    def _image_to_vec(self, X):
        return X.reshape(X.shape[0], self.feat_size * self.feat_size * self.channels)

    def fit(self, real_samples, labels):
        assert real_samples.ndim == 2
        assert labels.ndim == 2
        assert labels.shape[0] == real_samples.shape[0]
        assert real_samples.shape[1] == self.feature_volume, "Feature dimension should be {} but it is {}.".format(
            self.feature_volume, real_samples.shape[0])
        assert labels.shape[1] == 1

        self.observer["loss"] = 0
        self.observer["rec_loss"] = 0
        self.observer["reg_loss"] = 0
        # RESHAPING
        real_samples = self._vec_to_image(real_samples)

        # GPU RETYPING
        real_samples = self.retype(real_samples)
        labels = self.retype(labels)
        # PROCESSING DATA
        if self.mini_batch is None:
            self._back_propagation(real_samples, labels)
        else:
            indexes = self._xp.random.permutation(real_samples.shape[0])
            iternum = indexes.shape[0] // self.mini_batch
            if indexes.shape[0] % self.mini_batch > 0:
                iternum += 1
            for i in range(iternum):
                minibatch_indexes = indexes[self.mini_batch * i:self.mini_batch * (i + 1)]
                minibatch_real_samples = real_samples[minibatch_indexes, :]
                minibatch_labels = labels[minibatch_indexes, :]
                self._back_propagation(minibatch_real_samples, minibatch_labels)
            self.observer["loss"] /= float(iternum)
            self.observer["rec_loss"] /= float(iternum)
            self.observer["reg_loss"] /= float(iternum)
            # observations with different names for COMPATIBILIIY
            self.observer["cost"] = self.observer["loss"]
            self.observer["cost_decoder"] = self.observer["rec_loss"]
            self.observer["cost_encoder"] = self.observer["reg_loss"]

    def get_imitations(self, n):
        return self._image_to_vec(self.untype(
            self.model.decode(Variable(self.retype(self.latent_samples_generator(n, self.latent_size))))[0].data))

    def predict(self, samples, threshold=None, **argv):
        samples = self._vec_to_image(samples)
        if threshold is None:
            threshold = self.threshold
        evaluations = self.evaluate(samples)
        return (evaluations < threshold).reshape((-1, 1))

    def evaluate(self, samples):
        """
        Returns scalary real value for each given sample.
        :param samples: samples to evaluate
        :return: evaluation over samples, numpy array (1, samples.shape[0])
        """
        if len(samples) == 0:
            return []
        samples = self._vec_to_image(samples)
        samples = self.retype(samples)
        results = []
        indexes = self._xp.arange(0, samples.shape[0])
        iternum = indexes.shape[0] // self.mini_batch
        if indexes.shape[0] % self.mini_batch > 0:
            iternum += 1
        with no_backprop_mode():
            for i in range(iternum):
                minibatch_indexes = indexes[self.mini_batch * i:self.mini_batch * (i + 1)]
                minisamples = samples[minibatch_indexes, :]
                mu, ln_var = self.model.encode(minisamples)
                feature, extension = self.model.decode(mu)
                loss = 1 - F.sigmoid(extension)[:, 0]  # distance from one
                loss += F.reshape(F.sum_to(
                    F.squared_difference(minisamples, feature), shape=(len(minisamples), 1, 1, 1)
                ), (len(minisamples), )) / self.feature_volume
                results.append(loss.data)
        return self.untype(self._xp.concatenate(results))

    def save_model(self, pathh, prefix=""):
        self.model.save_model(pathh, prefix=prefix + "_{}".format(self.__class__.__name__))

    def save_args(self, pathh, prefix=""):
        arguments = [ar.strip() for ar in self._args_names.split(',')]
        dictionary = dict(zip(arguments, self._args))
        self._save_args(pathh, dictionary, prefix=prefix)

    def load_model(self, pathh, prefix=""):
        self.model.load_model(pathh, prefix=prefix + "_{}".format(self.__class__.__name__))

    @classmethod
    def create_from_args(cls, pathh, prefix="", units_clazz=None, overload_args=None):
        dictionary = cls._create_from_args(pathh, prefix=prefix, units_clazz=units_clazz)
        if overload_args is not None:
            for koa in overload_args:
                dictionary[koa] = overload_args[koa]
        return cls(feat_size=dictionary["feat_size"], hidden_size=dictionary["hidden_size"],
                   latent_size=dictionary["latent_size"], mini_batch=dictionary["mini_batch"],
                   learning_rate=dictionary["learning_rate"], gpu_on=dictionary["gpu_on"],
                   threshold=dictionary["threshold"],extended_dimension=dictionary["extended_dimension"],
                   threshold_extended=dictionary["threshold_extended"]
                   )

    @classmethod
    def build_random(cls, **kwargs):
        raise NotImplementedError("not implemented for this class")

    @staticmethod
    def empty_sample_set(dim):
        return np.empty((0, dim), dtype=np.float32)

    def __str__(self):
        arguments = [ar.strip() for ar in self._args_names.split(',')]
        dictionary = dict(zip(arguments, self._args))
        return "{} with arguments {}".format(self.__class__.__name__, dictionary)

    @staticmethod
    def normal_sampling(n, latent_space):
        return np.random.standard_normal((n, latent_space)).astype(dtype=np.float32)
