'''
Created on 30.03.2014

@author: hammen
'''

class Isotope(object):
    '''
    A simple isotope, currently unused and deprecated
    '''

    def __init__(self):
        '''
        Constructor
        '''
        #Fixed system properties
        self.name = "Scratch"
        self.mass = 1      #in u
        self.mass_d = 0.01   #error of mass in u
        self.I = 2.5         #nuclear Spin
        self.Jl = 0.5        #angular momentum of lower state
        self.Ju = 1.5        #angular momentum of upper state
        self.atomFreq = 100   #transition frequency of reference in MHz
        
        self.m = None   #reference of isomer
        
        #Fit parameters
        self.shift = 123     #expected shift (relative to atomFreq)
        self.Al = 1000        #expected A-factor of lower state
        self.Ar = 0.02        #expected A-factor of upper state
        self.Bl = 0        #expected B-factor of lower state
        self.Br = 197        #expected B-factor of upper state
        self.lorGam = 20   #expected lorentzian width
        self.fixLor = True
        self.gauSig = 30   #expected gaussian width
        self.fixGauss = True
        self.intScale = 40   #expected overall intensity
        
        #Fitting conditions
        self.fixedARat = False  #if True, only Al is used
        self.fixedBRat = False  #if True, only Bl is used        
        self.fixedInt = None          #List of relative intensities of hperfine components, set to None if free
        

        

        
        

        