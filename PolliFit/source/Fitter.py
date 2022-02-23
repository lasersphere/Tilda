"""
Created on 20.02.2022

@author: Patrick Mueller
"""


class Fitter:
    def __init__(self, model, meas, st):
        self.model = model

    def __call__(self, *args, **kwargs):
        """
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :returns: popt, pcov. Optimized parameters and an estimated of the covariances if possible. Compare 'curve_fit'.
        """
        pass

    def set_par(self, i, val):
        self.vals[i] = val

    def set_par_e(self, i, val):
        # TODO
        self.vals[i] = val

    def set_fix(self, i, fix):
        self.fixes[i] = fix

    def set_link(self, i, link):
        self.links[i] = link

    def reset(self):
        pass

    def pars_to_e(self):
        return zip(self.names, self.vals, self.fixes, self.links)
