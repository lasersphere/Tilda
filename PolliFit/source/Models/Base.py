"""
Created on 21.02.2022

@author: Patrick Mueller
"""

import numpy as np

import Tools


def args_ordered(args, order):
    return [args[i] for i in order]


def poly(x, *args):
    return np.sum([args[n] * x ** n for n in range(len(args))], axis=0)


class Model:
    def __init__(self, model=None):
        self.model = model
        self.type = 'Model'

    def __call__(self, x, *args, **kwargs):
        return self.evaluate(x, *self.update_args(*args), **kwargs)
    
    def evaluate(self, x, *args, **kwargs):  # Reimplement this function in subclasses (not evaluate).
        pass

    def _add_arg(self, name, val, fix, link):
        self.names.append(name)
        self.vals.append(val)
        self.fixes.append(fix)
        self.links.append(link)
        self.expressions.append('args[{}]'.format(self._index))

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
    def dx(self):
        return 0.1 if self.model is None else self.model.dx

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        if self._model is None:
            self.names, self.vals, self.fixes, self.links = [], [], [], []
            self.expressions = []
            self.p = {}
            self._index = 0
            self._size = 0
            self._error = ''
        else:
            self.names, self.vals, self.fixes, self.links = \
                self._model.names, self._model.vals, self._model.fixes, self._model.links
            self.expressions = self._model.expressions
            self.p = self._model.p
            self._index = len(self._model.names)
            self._size = len(self._model.names)
            self._error = self._model.error

    def get_pars(self):
        return zip(self.names, self.vals, self.fixes, self.links)

    def set_pars(self, pars):
        for i in range(len(self.vals)):
            self.vals[i] = pars[i][0]
            self.fixes[i] = pars[i][1]
            self.links[i] = pars[i][2]

    def set_vals(self, vals):
        for i in range(len(self.vals)):
            self.vals[i] = vals[i]

    def set_fixes(self, fixes):
        for i in range(len(self.fixes)):
            self.fixes[i] = fixes[i]

    def set_links(self, links):
        for i in range(len(self.links)):
            self.links[i] = links[i]
    
    def set_val(self, i, val):
        if isinstance(val, int) or isinstance(val, float):
            self.vals[i] = val
            self.set_vals(self.update_args(*self.vals))
    
    def set_fix(self, i, fix):
        if isinstance(fix, int) or isinstance(fix, float):
            fix = bool(fix)
            expr = 'args[{}]'.format(i)
        elif isinstance(fix, str):
            _fix = fix
            for j, name in enumerate(self.names):
                _fix = _fix.replace(name, 'eval(self.expressions[{}])'.format(j))
            expr = _fix
            try:
                eval(expr, {}, {'self': self, 'args': self.vals})
            except (ValueError, TypeError, SyntaxError, NameError) as e:
                print('Invalid expression for parameter \'{}\': {}. Got a {}.'.format(self.names[i], fix, repr(e)))
                return
        elif isinstance(fix, list):
            if len(fix) == 0:
                fix = [0, 1]
            elif len(fix) == 1:
                fix = [0, fix[0]]
            else:
                fix = fix[:2]
            expr = 'args[{}]'.format(i)
        else:
            return
        temp_expr = self.expressions[i]
        temp_fix = self.fixes[i]
        self.expressions[i] = compile(expr, '<string>', 'eval', optimize=2)  # Compile beforehand to save time.
        self.fixes[i] = fix
        try:
            self.set_vals(self.update_args(*self.vals))
        except RecursionError as e:
            print('Expressions form a loop. Got a {}.'.format(repr(e)))
            self.expressions[i] = temp_expr
            self.fixes[i] = temp_fix
    
    def set_link(self, i, link):
        if isinstance(link, int) or isinstance(link, float):
            self.links[i] = bool(link)

    def norm(self, *args, **kwargs):
        return self(0, *args, **kwargs)

    def min(self):
        return -1. if self.model is None else self.model.min()

    def max(self):
        return 1. if self.model is None else self.model.max()

    def intervals(self):
        return [[self.min(), self.max()]] if self.model is None else self.model.intervals()

    def x(self):
        return np.concatenate([np.arange(i[0], i[1], self.dx, dtype=float) for i in self.intervals()], axis=0)

    def fit_prepare(self):
        bounds = (-np.inf, np.inf)
        fixed = [fix for fix in self.fixes]
        b_lower = []
        b_upper = []
        _bounds = False
        for i, fix in enumerate(self.fixes):
            if isinstance(fix, bool):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
            elif isinstance(fix, list):
                _bounds = True
                b_lower.append(fix[0])
                b_upper.append(fix[1])
                fixed[i] = False
            elif isinstance(fix, str):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
                fixed[i] = True
                _fix = fix
            else:
                raise TypeError('The type {} of the element {} in self.fixes is not supported.'
                                .format(type(fix), fix))
        if _bounds:
            bounds = (b_lower, b_upper)
        return fixed, bounds

    def update_args(self, *args):
        return tuple(eval(expr, {}, {'self': self, 'args': args}) for expr in self.expressions)


class Empty(Model):
    def __init__(self):
        super().__init__(model=None)
        self.type = 'Empty'

    def evaluate(self, x, *args, **kwargs):
        return np.zeros_like(x)


class NPeak(Model):
    def __init__(self, model, n_peaks=1):
        super().__init__(model=model)
        self.type = 'NPeak'
        self.n_peaks = int(n_peaks)
        for n in range(self.n_peaks):
            self._add_arg('center{}'.format(n if n > 0 else ''), 0., False, False)
            self._add_arg('Int{}'.format(n), 1., n == 0, False)

    def evaluate(self, x, *args, **kwargs):
        return np.sum([args[self.model.size + 2 * n + 1]
                       * self.model.evaluate(x - args[self.model.size + 2 * n], *args[:self.model.size])
                       for n in range(self.n_peaks)], axis=0)

    def min(self):
        min_center = min(self.vals[self.p['center{}'.format(n if n > 0 else '')]] for n in range(self.n_peaks))
        return min_center + self.model.min()

    def max(self):
        max_center = max(self.vals[self.p['center{}'.format(n if n > 0 else '')]] for n in range(self.n_peaks))
        return max_center + self.model.max()

    def intervals(self):
        return Tools.merge_intervals([[i[0] + self.vals[self.model.size + 2 * n],
                                       i[1] + self.vals[self.model.size + 2 * n]]
                                      for i in self.model.intervals() for n in range(self.n_peaks)])


class Offset(Model):
    def __init__(self, model=None, x_cuts=None, offsets=None):
        """
        :param model: The model the offset will be added to. If None, the offset will be added to zero.
        :param x_cuts: x values where to cut the x-axis.
        :param offsets: A list of maximally considered polynomial orders for each slice.
         The list must have length len(x_cuts) + 1.
        """
        super().__init__(model=model)
        self.type = 'Offset'
        if x_cuts is None:
            x_cuts = []
        self.x_cuts = sorted(list(x_cuts))
        self.offsets = offsets
        if self.offsets is None:
            self.offsets = [0]
        if len(self.offsets) != len(self.x_cuts) + 1:
            raise ValueError('The parameter offset must be a list of size \'len(x_cuts) + 1\''
                             ' and contain the maximally considered polynomial order for each slice.')

        self.offset_map = []
        self.offset_masks = []
        self.update_on_call = True

        self.gen_offset_map()

    def evaluate(self, x, *args, **kwargs):
        if self._model is None:
            return self._offset(x, *args)
        return self.model.evaluate(x, *args[:self.model.size]) + self._offset(x, *args)

    def set_x_cuts(self, x_cuts):
        x_cuts = list(x_cuts)
        if len(x_cuts) != len(self.x_cuts):
            raise ValueError('\'x_cuts\' must not change its size.')
        self.x_cuts = sorted(list(x_cuts))

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
        for x0, x1 in zip([np.min(x) - 1., ] + self.x_cuts, self.x_cuts + [np.max(x) + 1., ]):
            x_mean = 0.5 * (x0 + x1)
            self.offset_masks.append(np.abs(x - x_mean) < x1 - x_mean)

    def guess_offset(self, x, y):
        for i, mask in enumerate(self.offset_masks):
            self.vals[self.p['off{}e0'.format(i)]] = 0.5 * (y[mask][0] + y[mask][-1])
            try:
                self.vals[self.p['off{}e1'.format(i)]] = (y[mask][-1] - y[mask][0]) / (x[mask][-1] - x[mask][0])
            except KeyError:
                return


class Amplifier(Model):
    def __init__(self, order=None):
        super().__init__(model=None)
        self.type = 'Amplifier'
        if order is None:
            order = 1
        self.order = order
        for n in range(order + 1):
            self._add_arg('a{}'.format(n), 1. if n == 1 else 0., False, False)
        self._min = -10
        self._max = 10

    def evaluate(self, x, *args, **kwargs):
        self._min = np.min(x)
        self._max = np.max(x)
        return poly(x, *args)

    @property
    def dx(self):
        return 1e-2

    def min(self):
        return self._min

    def max(self):
        return self._max


# class Linked(Model):
#     def __init__(self, model, definition):
#         super().__init__(model=model)
#         self.definition = definition
#
#         self.signal_map = []
#
#     """ Calculation """
#
#     def evaluate(self, x, *args, **kwargs):
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
