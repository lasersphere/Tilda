"""
Created on 21.02.2022

@author: Patrick Mueller
"""

import numpy as np


def args_ordered(args, order):
    return [args[i] for i in order]


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
    def __init__(self, model=None):
        self.model = model

    def __call__(self, x, *args, **kwargs):
        pass

    def _add_arg(self, name, val, fix, link):
        self.names.append(name)
        self.vals.append(val)
        self.fixes.append(fix)
        self.links.append(link)

        self.p[name] = self._index

        self._index += 1
        self._size += 1

    @property
    def size(self):
        """
        :returns: The number of parameters required by the model.
        """
        return self._size

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        if self._model is None:
            self.names, self.vals, self.fixes, self.links = [], [], [], []
            self.p = {}
            self._index = 0
            self._size = 0
        else:
            self.names, self.vals, self.fixes, self.links = \
                self._model.names, self._model.vals, self._model.fixes, self._model.links
            self.p = self._model.p
            self._index = len(self._model.names)
            self._size = len(self._model.names)

    def get_pars(self):
        return zip(self.names, self.vals, self.fixes, self.links)

    def set_vals(self, vals):
        for i in range(len(self.vals)):
            self.vals[i] = vals[i]

    def norm(self):
        return self(0)

    def min(self):
        return 0. if self.model is None else self.model.min()

    def max(self):
        return 0. if self.model is None else self.model.max()

    def x(self):
        return np.linspace(self.min(), self.max(), 1001, dtype=float)


class EmptyModel(Model):
    def __init__(self):
        super().__init__(model=None)

    def __call__(self, x, *args, **kwargs):
        return np.zeros_like(x)


class NPeak(Model):
    def __init__(self, model, n_peaks=1):
        super().__init__(model=model)
        self.n_peaks = int(n_peaks)
        for n in range(self.n_peaks):
            self._add_arg('center{}'.format(n if n > 0 else ''), 0., False, False)
            self._add_arg('Int{}'.format(n), 1., False, False)

    def __call__(self, x, *args, **kwargs):
        return np.sum([args[self.model.size + 2 * n + 1]
                       * self.model(x - args[self.model.size + 2 * n], *args[:self.model.size])
                       for n in range(self.n_peaks)], axis=0)

    def min(self):
        min_center = np.min([self.vals[self.p['center{}'.format(n if n > 0 else '')]] for n in range(self.n_peaks)])
        return min_center + self.model.min()

    def max(self):
        max_center = np.max([self.vals[self.p['center{}'.format(n if n > 0 else '')]] for n in range(self.n_peaks)])
        return max_center + self.model.max()


class Offset(Model):
    def __init__(self, model=None, x_cuts=None, offsets=None):
        super().__init__(model=model)
        if x_cuts is None:
            x_cuts = []
        self.x_cuts = sorted(x_cuts)
        self.offsets = offsets
        if self.offsets is None:
            self.offsets = [0]
        if len(self.offsets) != len(self.x_cuts) + 1:
            raise ValueError('The parameter offset must be a list of size \'len(x_cuts) + 1\''
                             ' and contain the maximum considered polynomial order for each slice.')

        self.offset_map = []
        self.offset_masks = []
        self.update_on_call = True

        self.gen_offset_map()

    def __call__(self, x, *args, **kwargs):
        if self._model is None:
            return self._offset(x, *args)
        return self.model(x, *args[:self.model.size]) + self._offset(x, *args)

    def _offset(self, x, *args):
        if self.update_on_call:
            self.gen_offset_masks(x)
        ret = np.zeros_like(x)
        for i, mask in enumerate(self.offset_masks):
            ret[mask] = poly(x[mask], *args_ordered(args, self.offset_map[i]))
        return ret

    def gen_offset_map(self):
        self.offset_map = []
        for i, n in enumerate(self.offsets):
            self.offset_map.append([])
            for k in range(n + 1):
                self.offset_map[-1].append(self._index)
                self._add_arg('off{}e{}'.format(i, k), 0., False, False)

    """ Preprocessing """

    def gen_offset_masks(self, x):
        self.offset_masks = []
        x_cut = -np.inf
        for x_cut in self.x_cuts:
            self.offset_masks.append(x < x_cut)
        self.offset_masks.append(x >= x_cut)


# class Linked(Model):
#     def __init__(self, model, definition):
#         super().__init__(model=model)
#         self.definition = definition
#
#         self.signal_map = []
#
#     """ Calculation """
#
#     def __call__(self, x, *args, **kwargs):
#         return np.array([self.definition.spec[i](x[i], *args_ordered(args, self.signal_map[i]), **kwargs)
#                          + self._offset(x, i, args) for i in range(self._size)], dtype=float)
#
#     def _offset(self, x, i, *args):
#         return np.concatenate(tuple(poly(x[i][s], *args_ordered(args, self.offset_map[i][j]))
#                                     for j, s in enumerate(self.offset_slices[i])), axis=0)
#
#     """ Model definition """
#
#     def dim(self):
#         """
#         :returns: The number of xy-axes required as input.
#         """
#         return 1
#
#     @property
#     def definition(self):
#         return self._definition
#
#     @definition.setter
#     def definition(self, value):
#         """
#         :param value: The new definition of the model.
#         :returns: None.
#         """
#         if not isinstance(value, Definition):
#             raise TypeError('The definition of a model must be a \'Definition\' object')
#         self._definition = value
#         self._size = self._definition.size
#         self._index = 0
#         self.names, self.vals, self.fixes, self.links = [], [], [], []
#
#     def gen_signal_map(self):
#         for i, spec in enumerate(self._definition.spec):
#             self.signal_map.append([])
#             for name, val, fix, link in zip(spec.names, spec.vals, spec.fixes, spec.links):
#                 self._add_arg('{}__{}'.format(name, i), val, fix, link)
#                 self.signal_map[-1].append(self._index)
#                 self._index += 1
