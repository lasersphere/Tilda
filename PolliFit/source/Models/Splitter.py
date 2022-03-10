"""
Created on 02.03.2022

@author: Patrick Mueller
"""

import Physics as Ph
from Models.Base import *


def gen_splitter_models(config, iso):
    if config['qi'] and config['hf_mixing']:
        pass
    elif config['qi']:
        pass
    elif config['hf_mixing']:
        pass
    else:
        return [Hyperfine], [(iso.I, iso.Jl, iso.Ju)]
    raise ValueError('Specified splitter model not available.')


class Splitter(Model):
    def __init__(self, model, i, j_l, j_u):
        super().__init__(model=model)
        self.type = 'Splitter'
        self.i = i
        self.j_l = j_l
        self.j_u = j_u

    def racah(self):
        pass


class Hyperfine(Splitter):
    def __init__(self, model, i, j_l, j_u):
        super().__init__(model, i, j_l, j_u)
        self.type = 'Hyperfine'
        self.transitions = Ph.HFTrans(self.i, self.j_l, self.j_u)
        self.intensities = Ph.HFInt(self.i, self.j_l, self.j_u, self.transitions)
        self.indices = []

        self._add_arg('Al', 0, self.transitions[0][2] == 0, False)
        self._add_arg('Bl', 0, self.transitions[0][3] == 0, False)
        self._add_arg('Au', 0, self.transitions[0][4] == 0, False)
        self._add_arg('Bu', 0, self.transitions[0][5] == 0, False)
        for (f_l, f_u, *_), intensity in zip(self.transitions, self.intensities):
            self.indices.append(self._index)
            self._add_arg('IntFl{}Fu{}'.format(f_l, f_u), intensity, False, False)

    def evaluate(self, x, *args, **kwargs):
        al, bl, au, bu = tuple(args[self.model.size + i] for i in range(4))
        return np.sum([args[i] * self.model.evaluate(x - Ph.HFShift(al, bl, au, bu, *c), *args, **kwargs)
                       for i, (_, _, *c) in zip(self.indices, self.transitions)], axis=0)

    def min(self):
        al, bl, au, bu = tuple(self.vals[self.model.size + i] for i in range(4))
        return self.model.min() + min(Ph.HFShift(al, bl, au, bu, *c) for _, _, *c in self.transitions)

    def max(self):
        al, bl, au, bu = tuple(self.vals[self.model.size + i] for i in range(4))
        return self.model.max() + max(Ph.HFShift(al, bl, au, bu, *c) for _, _, *c in self.transitions)

    def intervals(self):
        al, bl, au, bu = tuple(self.vals[self.model.size + i] for i in range(4))
        shifts = [Ph.HFShift(al, bl, au, bu, *c) for _, _, *c in self.transitions]
        return Tools.merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts])

    def racah(self):
        for i, intensity in zip(self.indices, self.intensities):
            self.vals[i] = intensity
