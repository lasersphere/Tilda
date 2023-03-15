'''
Created on 27.01.2017

@author: schmid
'''

from Tilda.PolliFit import Physics


class LorentzQI(object):
    '''
    Implementation of a lorentzian profile object using lorentzian peaks and their interferences
    
    The peak height is normalized to a single lorentzian
    Gamma is the half width half maximum
    '''

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 1
        self.pGam = 0
        self.recalc([iso.shape.get('lor', iso.shape.get('gamma', 0.0))])


    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''
        return Physics.lorentz(x, 0, p[self.pGam]) / self.norm

    
    def evaluateQI(self, x, p, j, lineSplit):
        '''Return the value of the hyperfine structure at point x / MHz'''
        pj = -1
        for i in range(len(lineSplit)):
            if j == lineSplit[i]:
                pj = i

        if pj == -1:
            return 0
        else:
            if self.iso.Ju != 0.5:
                pjc = pj - 2 if pj%3 == 2 else pj + 1

                return Physics.lorentzQI(x, 0, lineSplit[pjc] - lineSplit[pj], p[self.pGam]) / self.norm / 2

            else:
                print('Evaluation for Ju = 0,5 is not implemented!')

    
    def recalc(self, p):
        '''Recalculate the norm factor'''
        self.norm = Physics.lorentz(0, 0, p[self.pGam])
    
    
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        return -5 * p[self.pGam]
    
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        return 5 * p[self.pGam]
    
    
    def getPars(self, pos = 0):
        '''Return list of initial parameters and initialize positions'''
        self.pGam = pos
        
        return [self.iso.shape.get('lor', self.iso.shape.get('gamma', 0.0))]
    
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['gamma']
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape.get('lor', self.iso.fixShape.get('gamma', False))]
        