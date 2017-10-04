# coding=utf-8
import numpy as np


class RandomVariable:
    def __init__(self, array):
        self.array = np.array(array)
        self.mean = self.array.mean(axis=0)
        self.std = self.array.std(axis=0)
        self.se = self.std / np.sqrt(len(self.array))

    def __gt__(self, other):
        return self.mean > other.mean

    def __ge__(self, other):
        return self.mean >= other.mean

    def __lt__(self, other):
        return self.mean < other.mean

    def __le__(self, other):
        return self.mean <= other.mean

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.std

    def __format__(self, format_spec):
        tmp = [u'{:{spec}} ± {:{spec}}'.format(mean, std, spec=format_spec)
               for mean, std in zip(self.mean.flat, self.std.flat)]
        if len(tmp) == 1:
            return tmp[0]
        else:
            return np.char.array(tmp).reshape(self.mean.shape)

    def to_string(self, decimal_places):
        format_spec='.'+str(decimal_places)+'f'
        return self.__format__(format_spec)