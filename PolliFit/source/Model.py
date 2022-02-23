"""
Created on 21.02.2022

@author: Patrick Mueller
"""

import numpy as np

from Spectra.Spectrum import Spectrum


def args_ordered(args, p0, order):
    return [p0[-i - 1] if i < 0 else args[i] for i in order if i is not None]


def poly(x, *args):
    return np.sum([args[n] * x ** n for n in range(len(args))], axis=0)


class Definition:
    def __init__(self, definitions):
        self.definitions = list(definitions)
        self.spec = [d['spec'] for d in self.definitions]
        self._size = len(self.definitions)

    def __iter__(self):
        for definition in self.definitions:
            yield definition

    def __getitem__(self, key: int):
        return self.definitions[key]

    @property
    def size(self):
        return self._size


class Model:
    def __init__(self, definition=None):
        self._size = 1  # Number of xy-axes required as input.

        self.definition = definition

        self.names = []
        self.vals = []
        self.fixes = []
        self.links = []
        self.p0 = []

        self.signal_map = []

        self.offset_map = []
        self.offset_slices = [[slice(0, None, 1)]]

        self._index = 0

    """ Calculation """

    def __call__(self, x, *args, **kwargs):
        return np.array([self.definition.spec[i](x[i], *args_ordered(args, self.p0, self.signal_map[i]), **kwargs)
                         + self._offset(x, i, args) for i in range(self._size)], dtype=float)

    def _offset(self, x, i, *args):
        return np.array([poly(x[i][s], *args_ordered(args, self.p0, self.offset_map[i][j]))
                         for j, s in enumerate(self.offset_slices[i])], dtype=float)

    """ Model definition """

    def add_arg(self, i, name, fix, link, p0):
        pass

    @property
    def size(self):
        return self._size

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self, value):
        """

        :param value: The new definition of the model.
        :returns: None.
        """
        if not isinstance(value, Definition):
            raise TypeError('The definition of a model must be a \'Definition\' object')
        self._definition = value
        self._size = self.definition.size
        self.gen_signal_map()
        self.gen_offset_map()

    def gen_signal_map(self):
        pass

    def gen_offset_map(self):
        self.offset_map = []
        for i, d in enumerate(self.definition):
            self.offset_map.append([])
            for j, p0 in enumerate(d['offset']):
                for k, _p0 in enumerate(p0):
                    self.offset_map.append([self._index if _p0 != 0 and ])
                    self.add_arg(self._index, 'off{}e{}__{}'.format(j, k, i), False, False, _p0)
                    self._index += 1

    """ Prints """
