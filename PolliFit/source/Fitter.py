"""
Created on 20.02.2022

@author: Patrick Mueller
"""


class Fitter:
    def __init__(self):
        self.names = []
        self.vals = []
        self.fixes = []
        self.links = []

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
