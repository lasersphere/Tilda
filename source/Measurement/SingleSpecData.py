'''
Created on 23.03.2014

@author: hammen
'''

class SingleSpecData(object):
    '''
    This object contains only x and y data for a single spectrum. It is handed over to the fitting routine.
    '''

    def __init__(self, x, cts, err):
        '''
        Constructor
        '''     
        self.x = x
        self.y = cts
        self.err = err