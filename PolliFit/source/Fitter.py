"""
Created on 20.02.2022

@author: Patrick Mueller
"""


class ModelFitter:
    def __init__(self, model, meas, st):
        self.model = model
        self.meas = meas
        self.st = st

    def get_pars(self):
        return self.model.get_pars()
