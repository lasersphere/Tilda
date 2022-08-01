"""
Created on 02.03.2022

@author: Patrick Mueller
"""

from string import ascii_uppercase
import numpy as np

import Physics as Ph
from Models.Base import *


def gen_splitter_model(config, iso):
    if config['qi'] and (config['hf_config']['enabled_l'] or config['hf_config']['enabled_u']):
        pass
    elif config['qi']:
        pass
    elif config['hf_config']['enabled_l'] or config['hf_config']['enabled_u']:
        return HyperfineMixed, (iso.I, iso.Jl, iso.Ju, iso.name, config['hf_config'])
    else:
        return Hyperfine, (iso.I, iso.Jl, iso.Ju, iso.name)
    raise ValueError('Specified splitter model not available.')


def gen_splitter_models(config, iso):
    splitter, _args = gen_splitter_model(config, iso)
    args = [_args, ]
    _iso = iso.m
    while _iso is not None:
        _, _args = gen_splitter_model(config, _iso)
        args.append(_args)
        _iso = _iso.m
    return splitter, args


class Splitter(Model):
    def __init__(self, model, i, j_l, j_u, name):
        super().__init__(model=model)
        self.type = 'Splitter'

        self.i = i
        self.j_l = j_l
        self.j_u = j_u
        self.name = name

        self.racah_indices = []
        self.racah_intensities = []

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class SplitterSummed(Summed):
    def __init__(self, splitter_models):
        if any(not isinstance(model, Splitter) for model in splitter_models):
            raise TypeError('All models passed to \'SplitterSummed\' must have type \'Splitter\'.')
        super().__init__(splitter_models, labels=['({})'.format(model.name) for model in splitter_models]
                         if len(splitter_models) > 1 else None)

    def racah(self):
        i0 = 0
        for model in self.models:
            for i, intensity in zip(model.racah_indices, model.racah_intensities):
                self.set_val(i0 + i, intensity, force=True)
            i0 += model.size
        self.set_vals(self.vals, force=True)


class Hyperfine(Splitter):
    def __init__(self, model, i, j_l, j_u, name):
        super().__init__(model, i, j_l, j_u, name)
        self.type = 'Hyperfine'

        self.transitions = Ph.HFTrans(self.i, self.j_l, self.j_u, old=False)
        self.racah_intensities = Ph.HFInt(self.i, self.j_l, self.j_u, self.transitions, old=False)

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg('{}l'.format(ascii_uppercase[i]), 0., False, False)
        for i in range(self.n_u):
            self._add_arg('{}u'.format(ascii_uppercase[i]), 0., False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg('int({}, {})'.format(t[0][0], t[0][1]), intensity, i == 0, False)

    def evaluate(self, x, *args, **kwargs):
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum([args[i] * self.model.evaluate(x - Ph.HFShift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                       for i, t in zip(self.racah_indices, self.transitions)], axis=0)

    def min(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def max(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def intervals(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return Tools.merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts])

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class HyperfineMixed(Splitter):
    def __init__(self, model, i, j_l, j_u, name, config):
        super().__init__(model, i, j_l, j_u, name)
        self.type = 'HyperfineMixed'
        self.config = config

        self.transitions = Ph.HFTrans(self.i, self.j_l, self.j_u, old=False)
        self.racah_intensities = Ph.HFInt(self.i, self.j_l, self.j_u, self.transitions, old=False)

        self.T_l = None
        self.f_l = None
        self.W_l = None
        self.n_l = len(self.transitions[0][1])
        if self.config['enabled_l']:
            self.T_l = np.array(self.config['Tl'])
            self.f_l = np.array(self.config['fl']).flatten()
            self.f_l -= self.f_l[0]
            for i, jl in enumerate(self.config['Jl']):
                self._add_arg('FS_l{}({})'.format(i, jl), 0., False, False)
            self.n_l = len(self.config['Jl'] - 1)
            self.W_l = [np.array([[(-1) ** (self.i + ji + f[0]) * Ph.sixJ(self.i, ji, f[0], jj, self.i, 1)
                                   for jj in self.config['Jl']] for ji in self.config['Jl']]) * self.T_l
                        for f, *r in self.transitions]
        else:
            for i in range(self.n_l):
                self._add_arg('{}l'.format(ascii_uppercase[i]), 0., False, False)

        self.T_u = None
        self.f_u = None
        self.j_lists = None
        self.n_u = len(self.transitions[0][2])
        if self.config['enabled_u']:
            self.T_u = np.array(self.config['Tu'])
            self.f_u = np.array(self.config['fu']).flatten()
            self.f_u -= self.f_u[0]
            for i, ju in enumerate(self.config['Ju']):
                self._add_arg('FS_u{}({})'.format(i, ju), 0., False, False)
            self.n_u = len(self.config['Ju'])
            self.j_lists = [np.array([i for i, j in enumerate(self.config['Ju'])
                                      if np.abs(self.i - j) - 0.1 < f[1] < self.i + j + 0.1])
                            for f, *r in self.transitions]
            self.W_u = [np.array([[(-1) ** (self.i + self.config['Ju'][i] + f[1])
                                   * Ph.sixJ(self.i, self.config['Ju'][i], f[1], self.config['Ju'][j], self.i, 1)
                                   for j in j_list] for i in j_list]) * self.T_u[j_list] * self.config['mu']
                        for j_list, (f, *r) in zip(self.j_lists, self.transitions)]
        else:
            for i in range(self.n_u):
                self._add_arg('{}u'.format(ascii_uppercase[i]), 0., False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg('int({}, {})'.format(t[0][0], t[0][1]), intensity, i == 0, False)

    def evaluate(self, x, *args, **kwargs):
        np.zeros_like(x)
        # if self.config['enabled_u']:
        #     x_u = [np.diag([0.] + [args[self.model.size + self.n_l + i] for i in j_list]) + w
        #            for j_list, w in (self.j_lists, self.W_u)]
        #     x_u = [np.linalg.eigh(w) for w in x_u]
        # else:
        #     const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        #     x_u = [Ph.HFShift([], const_u, [], t[2]) for t in self.transitions]
        #
        # const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        # return np.sum([args[i] * self.model.evaluate(x - (_x_u - _x_l), *args, **kwargs)
        #                for i, _x_l, _x_u in zip(self.racah_indices, x_l, x_u)], axis=0)

    def min(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def max(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions)

    def intervals(self):
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [Ph.HFShift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return Tools.merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts])

    def racah(self):
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity
