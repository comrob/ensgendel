import numpy as np
from models.ensgendel.classifier import ClassifierEnsemble


class AddDelEnsembleClassifier(ClassifierEnsemble):
    EMPTY_BOOL = np.empty((0,), dtype=np.bool)
    def __init__(self, units, threshold, max_updates, min_updates, subtract_epsilon, maximizing_units, uncover_off, gpu_on, max_gen_samples):
        super(AddDelEnsembleClassifier, self).__init__(units, maximizing_units, gpu_on)
        self._args = threshold, max_updates, min_updates, subtract_epsilon, maximizing_units, uncover_off, gpu_on, max_gen_samples
        self._sgnt = "threshold, max_updates, min_updates, subtract_epsilon, maximizing_units, uncover_off, gpu_on, max_gen_samples"
        self.threshold = threshold
        self.max_updates = max_updates
        self.min_updates = min_updates
        self.subtract_epsilon = subtract_epsilon
        self.max_gen_samples = max_gen_samples
        self.add_acceptance = 0.2
        self.del_acceptance = 0.2
        self._prev_cost_add = 0
        self._prev_cost_del = 0
        self._uncover_off = uncover_off

    def purge_optimizers(self):
        for unit in self.units:
            unit.optimiser = None
            unit.remodel()

    def cover(self, unit_index, samples, outside_samples, sampling_on=True):
        if sampling_on:
            pos_generated = self.samples_provider.get_positive_samples(unit_index)
            neg_generated = self.samples_provider.get_negative_samples(unit_index)
            pos_indx_generated = np.random.permutation(pos_generated.shape[0])[:self.max_gen_samples]
            neg_indx_generated = np.random.permutation(neg_generated.shape[0])[:self.max_gen_samples]
            pos_generated = pos_generated[pos_indx_generated, :]
            neg_generated = neg_generated[neg_indx_generated, :]
            if not self._uncover_off:
                pos_generated = subtract_eps_neighbourhoods(pos_generated, outside_samples, self.subtract_epsilon)
                neg_generated = subtract_eps_neighbourhoods(neg_generated, samples, self.subtract_epsilon)
            pos_samples = np.concatenate((samples, pos_generated))
            neg_samples = np.concatenate((outside_samples, neg_generated))
        else:
            pos_generated = np.empty((0, samples.shape[1]), dtype=np.float32)
            neg_generated = np.empty((0, samples.shape[1]), dtype=np.float32)
            pos_samples = samples
            neg_samples = outside_samples
        for i in range(self.max_updates):

            pos_inp_samples_in = self.in_wrap_of(unit_index, samples)
            neg_inp_samples_in = self.in_wrap_of(unit_index, outside_samples)
            pos_gen_samples_in = self.in_wrap_of(unit_index, pos_generated)
            neg_gen_samples_in = self.in_wrap_of(unit_index, neg_generated)


            rate_pos_inp_samples_in = np.sum(pos_inp_samples_in)/float(len(samples)) if len(samples) > 0 else 0
            rate_neg_inp_samples_in = np.sum(neg_inp_samples_in)/float(len(outside_samples)) if len(outside_samples) > 0 else 0
            rate_pos_gen_samples_in = np.sum(pos_gen_samples_in)/float(len(pos_generated)) if len(pos_generated) > 0 else 0
            rate_neg_gen_samples_in = np.sum(neg_gen_samples_in)/float(len(neg_generated)) if len(neg_generated) > 0 else 0



            print("ADD {}-epoch:{}; pos:{}/{}|{}/{}({}%|{}%);   neg:{}/{}|{}/{}({}%|{}%);   J:{}".format(
                unit_index, i,
                np.sum(pos_inp_samples_in), len(samples),np.sum(pos_gen_samples_in), len(pos_generated),
                int(100 * rate_pos_inp_samples_in), int(100 * rate_pos_gen_samples_in),
                np.sum(neg_inp_samples_in), len(outside_samples), np.sum(neg_gen_samples_in), len(neg_generated),
                int(100 * rate_neg_inp_samples_in), int(100 * rate_neg_gen_samples_in),
                self._prev_cost_add
            ))

            pos_gen_notaccept = (1-rate_pos_gen_samples_in) > self.add_acceptance if len(pos_generated) > 0 else False
            pos_inp_notaccept = (1-rate_pos_inp_samples_in) > self.add_acceptance
            neg_gen_notaccept = rate_neg_gen_samples_in > self.add_acceptance if len(neg_generated) > 0 else False
            neg_inp_notaccept = rate_neg_inp_samples_in > self.add_acceptance

            if i < self.min_updates or pos_gen_notaccept or pos_inp_notaccept or neg_gen_notaccept or neg_inp_notaccept:
                posneg_samples, posneg_labels = self.create_posneg_batch(
                    # pos_samples[~pos_samples_in,:], neg_samples[neg_samples_in,:]
                    pos_samples, neg_samples
                )
                self.units[unit_index].fit(posneg_samples, posneg_labels)
                # print "ep{} J:{}".format(i, self.units[unit_index].observer["cost"])
                self._prev_cost_add = self.units[unit_index].observer["cost"]
                self.observer["cover_update_costs"][unit_index].append(self._prev_cost_add)
                self.observer["cover_updates_number"][unit_index] += 1
            else:
                break
        self.observer["cover_costs"][unit_index] = sum(self.observer["cover_update_costs"][unit_index]) / float(
            len(self.observer["cover_update_costs"][unit_index]) + 0.001)

    def uncover(self, unit_index, samples, inside_samples, sampling_on=True):
        if sampling_on:
            pos_generated = self.samples_provider.get_positive_samples(unit_index)
            neg_generated = self.samples_provider.get_negative_samples(unit_index)
            pos_indx_generated = np.random.permutation(pos_generated.shape[0])[:self.max_gen_samples]
            neg_indx_generated = np.random.permutation(neg_generated.shape[0])[:self.max_gen_samples]
            pos_generated = pos_generated[pos_indx_generated, :]
            pos_generated = subtract_eps_neighbourhoods(pos_generated, samples, self.subtract_epsilon)
            neg_generated = neg_generated[neg_indx_generated, :]
            neg_generated = subtract_eps_neighbourhoods(neg_generated, inside_samples, self.subtract_epsilon)
            pos_samples = np.concatenate((pos_generated, inside_samples))
            neg_samples = np.concatenate((neg_generated, samples))
        else:
            pos_generated = np.empty((0, samples.shape[1]), dtype=np.float32)
            neg_generated = np.empty((0, samples.shape[1]), dtype=np.float32)
            pos_samples = inside_samples
            neg_samples = samples
        for i in range(self.max_updates):
            pos_inp_samples_in = self.in_wrap_of(unit_index, inside_samples)
            neg_inp_samples_in = self.in_wrap_of(unit_index, samples)
            pos_gen_samples_in = self.in_wrap_of(unit_index, pos_generated)
            neg_gen_samples_in = self.in_wrap_of(unit_index, neg_generated)


            rate_pos_inp_samples_in = np.sum(pos_inp_samples_in)/float(len(inside_samples)) if len(inside_samples) > 0 else 0
            rate_neg_inp_samples_in = np.sum(neg_inp_samples_in)/float(len(samples)) if len(samples) > 0 else 0
            rate_pos_gen_samples_in = np.sum(pos_gen_samples_in)/float(len(pos_generated)) if len(pos_generated) > 0 else 0
            rate_neg_gen_samples_in = np.sum(neg_gen_samples_in)/float(len(neg_generated)) if len(neg_generated) > 0 else 0
            print("DEL {}-epoch:{}; pos:{}/{}|{}/{}({}%|{}%);   neg:{}/{}|{}/{}({}%|{}%);   J:{}".format(
                unit_index, i,
                np.sum(pos_inp_samples_in), len(inside_samples),np.sum(pos_gen_samples_in), len(pos_generated),
                int(100 * rate_pos_inp_samples_in), int(100 * rate_pos_gen_samples_in),
                np.sum(neg_inp_samples_in), len(samples), np.sum(neg_gen_samples_in), len(neg_generated),
                int(100 * rate_neg_inp_samples_in), int(100 * rate_neg_gen_samples_in),
                self._prev_cost_del
            ))

            pos_gen_notaccept = (1-rate_pos_gen_samples_in) > self.del_acceptance if len(pos_generated) > 0 else False
            pos_inp_notaccept = (1-rate_pos_inp_samples_in) > self.del_acceptance if len(inside_samples) > 0 else False
            neg_gen_notaccept = rate_neg_gen_samples_in > self.del_acceptance if len(neg_generated) > 0 else False
            neg_inp_notaccept = rate_neg_inp_samples_in > self.del_acceptance if len(samples) > 0 else False

            if i < self.min_updates or pos_gen_notaccept or pos_inp_notaccept or neg_gen_notaccept or neg_inp_notaccept:
                posneg_samples, posneg_labels = self.create_posneg_batch(
                    # pos_samples[~pos_samples_in,:], neg_samples[neg_samples_in,:]
                    pos_samples, neg_samples
                )
                self.units[unit_index].fit(posneg_samples, posneg_labels)
                # print "ep{} J:{}".format(i, self.units[unit_index].observer["cost"])
                self._prev_cost_del = self.units[unit_index].observer["cost"]
                self.observer["tmp_update_costs"][unit_index].append(self._prev_cost_del)
                self.observer["tmp_updates_number"][unit_index] += 1
            else:
                break

        # self.observer["tmp_costs"][unit_index] += sum(self.observer["tmp_update_costs"][unit_index]) / float(
        #     len(self.observer["tmp_update_costs"][unit_index]) + 0.001)

    def fit(self, batch_sample, batch_labels, sampling_on=True, **kwargs):
        self._init_observer()
        label_sels = self.get_label_indexes(batch_labels)
        for i in range(len(self.units)):
            added_samples = batch_sample[label_sels[i], :]
            if len(added_samples) == 0:
                continue
            outside_samples = np.concatenate([batch_sample[label_sels[j], :] for j in range(len(self.units)) if j != i])
            print("TRAIN {}<-{}.".format(i, len(added_samples)))
            if not self._uncover_off:
                for k in range(len(self.units)):
                    if i == k:
                        continue
                    if self.in_wrap_of(k, added_samples).any():
                        self._init_tmp_observer(k)
                        inside_samples = batch_sample[label_sels[k], :]
                        self.uncover(k, added_samples, inside_samples, sampling_on=sampling_on)
                self._record_uncovers(i)
            self.cover(i, added_samples, outside_samples, sampling_on=sampling_on)
        self.observer["cost"] = sum(self.observer["cover_costs"]) / float(len(self.units))
        self.observer["total_updates"] = sum(self.observer["cover_updates_number"])

    def _record_uncovers(self, unit_id):
        # TODO implement this
        self.observer["uncover_cost"][unit_id] = 0
        self.observer["uncover_total_updates"][unit_id] = 0

    def _init_tmp_observer(self, unit_id):
        self.observer["tmp_update_costs"] = self.observer["uncover_update_costs"][unit_id]
        self.observer["tmp_updates_number"] = self.observer["uncover_updates_number"][unit_id]
        self.observer["tmp_costs"] = self.observer["uncover_costs"][unit_id]

    def _init_observer(self):
        self.observer["cover_update_costs"] = [[] for i in range(len(self.units))]
        self.observer["cover_updates_number"] = [0] * len(self.units)
        self.observer["cover_costs"] = [0] * len(self.units)
        self.observer["cost"] = 0
        self.observer["total_updates"] = 0

        self.observer["tmp_update_costs"] = [[] for i in range(len(self.units))]
        self.observer["tmp_updates_number"] = [0] * len(self.units)
        self.observer["tmp_costs"] = [0] * len(self.units)

        self.observer["uncover_update_costs"] = [[[] for j in range(len(self.units))] for i in range(len(self.units))]
        self.observer["uncover_updates_number"] = [[0] * len(self.units) for i in range(len(self.units))]
        self.observer["uncover_costs"] = [[[0] * len(self.units)] for i in range(len(self.units))]
        self.observer["uncover_cost"] = [0] * len(self.units)
        self.observer["uncover_total_updates"] = [0] * len(self.units)

    def in_wrap_of(self, unit_index, samples, shift=0.0):
        if len(samples) == 0:
            return self.EMPTY_BOOL
        evaluated = self.units[unit_index].evaluate(samples)
        # print "{},{}".format(len(evaluated), np.sum(self.cmp(evaluated, self.threshold)))
        return self.cmp(evaluated, self.threshold + shift)

    def predict(self, samples, pedantic=False):
        embedding = self.embed(samples)
        predictions = self.argbest(self.embed(samples), axis=1)
        if pedantic:
            outside_of_wrap_mask = ~self.cmp(self.best(embedding, axis=1), self.threshold)
            predictions[outside_of_wrap_mask] = -1
        return predictions.reshape(-1, 1)

    def save_args(self, pathh, prefix=""):
        arguments = [ar.strip() for ar in self._sgnt.split(',')]
        dictionary = dict(zip(arguments, self._args))
        self._save_args(pathh, dictionary, prefix=prefix)

    def __str__(self):
        arguments = [ar.strip() for ar in self._sgnt.split(',')]
        dictionary = dict(zip(arguments, self._args))
        return "{} with arguments {} and unit: {}".format(self.__class__.__name__, dictionary, str(self.units[0]))

    @classmethod
    def create_from_args(cls, pathh, prefix="", units_clazz=None, overload_args=None):
        units, dictionary = cls._create_from_args(pathh, prefix=prefix, units_clazz=units_clazz,
                                                  overload_args=overload_args)
        if "uncover_off" not in dictionary:  # For backward compatibility
            dictionary["uncover_off"] = False
        if "max_gen_samples" not in dictionary:
            dictionary["max_gen_samples"] = 200
        return cls(units, threshold=dictionary["threshold"], max_updates=dictionary["max_updates"],
                   min_updates=dictionary["min_updates"], subtract_epsilon=dictionary["subtract_epsilon"],
                   maximizing_units=dictionary["maximizing_units"], gpu_on=dictionary["gpu_on"],
                   uncover_off=dictionary["uncover_off"], max_gen_samples=dictionary["max_gen_samples"])



def subtract_eps_neighbourhoods(points, centers, eps):
    ret = points
    for i in range(centers.shape[0]):#TODO: metrika musi byt volitelna
        ret = ret[np.sqrt(np.sum(np.square(ret - centers[i, :]), axis=1)) > eps, :]
    return ret
