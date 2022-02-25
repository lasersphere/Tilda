"""
Created on 20.02.2022

@author: Patrick Mueller
"""


import numpy as np


class ModelFitter:
    def __init__(self, model, meas, st, iso):
        self.model = model
        self.meas = meas
        self.st = st
        self.iso = iso

    def get_pars(self):
        return self.model.get_pars()
