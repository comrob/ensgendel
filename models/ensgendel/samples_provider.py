import numpy as np

class SamplesProvider(object):
    def __init__(self, units):
        self._units = units
        self.unit_number = len(units)

    def get_positive_samples(self, unit_id):
        pass

    def get_negative_samples(self, unit_id):
        pass


class DebugSampleProvider(SamplesProvider):
    def __init__(self, units, task1_samples):
        super(DebugSampleProvider, self).__init__(units)
        self.samples_collection = task1_samples

    def get_positive_samples(self, unit_id):
        return self.samples_collection[unit_id]

    def get_negative_samples(self, unit_id):
        return np.concatenate([self.samples_collection[i] for i in range(len(self._units)) if i != unit_id])


class CompactGanSampleProvider(SamplesProvider):
    def __init__(self, units):
        super(CompactGanSampleProvider, self).__init__(units)
        self._imitation_n = 300
        self._trials = 10
        self._dirty = [True] * len(units)
        self._samples = [[] for i in range(len(units))]

    def _extract_from_gan(self, unit_id):
        aggr_imits = []
        imn = 0
        for i in range(self._trials):
            imits = self._units[unit_id].get_imitations(self._imitation_n)
            imits_filter = self._units[unit_id].predict(imits, pedantic=True)
            good_imits = imits[imits_filter[:, 0], :]
            aggr_imits.append(good_imits)
            imn += len(good_imits)
            if imn >= self._imitation_n:
                break

        return np.concatenate(aggr_imits)

    def _get_positive_samples(self, unit_id, called=True):
        if self._dirty[unit_id]:
            self._samples[unit_id] = self._extract_from_gan(unit_id)
            self._dirty[unit_id] = False
        return self._samples[unit_id]

    def get_positive_samples(self, unit_id):
        return self._get_positive_samples(unit_id, called=False)

    def get_negative_samples(self, unit_id):
        return np.concatenate(
            [self._get_positive_samples(i, called=False) for i in range(len(self._units)) if i != unit_id])
