"""
Created on 02.03.2022

@author: Patrick Mueller
"""

from Models.Base import *


def gen_splitter_model(config):
    if config['qi'] and config['hf_mixing']:
        pass
    elif config['qi']:
        pass
    elif config['hf_mixing']:
        pass
    else:
        return Hyperfine
    raise ValueError('Specified splitter model not available.')


class Hyperfine(Model):
    def __init__(self, model):
        super().__init__(model=model)

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
