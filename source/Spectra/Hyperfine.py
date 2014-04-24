'''
Created on 23.03.2014

@author: hammen
'''

import math
import Physics

class Hyperfine(object):
    '''
    classdocs
    '''


    def __init__(self, iso, shape):
        '''
        Constructor
        '''
        self.iso = iso
        self.shape = shape
        
        self.transitions = Physics.HFTrans(self.iso.I, self.iso.Ju, self.iso.Jl)
        self.linePos = Physics.HFLinePos(self.Au, self.Bu, self.Al, self.Bl, self.transitions)
        
    
    def initPars(self):
        pass
        
    def getLeftEdge(self):
        pass
    
    def getRightEdge(self):
        pass
        


        
    