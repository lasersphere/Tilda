'''
Created on 29.04.2014

@author: hammen
'''

import matplotlib.pyplot as plt
import numpy as np

class MPLPlotter(object):
    '''
    Maybe something like a pyplot wrapper. Deprecated
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    
    def printSpec(self, spec, par):
        x = np.linspace(spec.leftEdge(), spec.rightEdge(), 10000)
        y = np.fromiter((spec.evaluate([m], par) for m in x), np.float32)

        plt.plot(x, y)
        plt.ylabel('Counts / a.u.')
        plt.xlabel('Frequency / MHz')
        
        plt.draw()
        