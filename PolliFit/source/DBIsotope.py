'''
Created on 25.04.2014

@author: hammen
'''

import Physics
import TildaTools as TiTS

import sqlite3


class DBIsotope(object):
    '''
    A sqlite database driven version fo the isotope object
    '''

    def __init__(self, db, iso, isovar = '', lineVar = ''):
        '''Load relevant values of isotope name from database file'''
        print('iso: ' + str(iso))
        print('isovar: ' + str(isovar))
        print("Loading", lineVar, "line of", iso + isovar)
        #sqlite3.register_converter("BOOL", lambda v: bool(int(v)))

        data = TiTS.select_from_db(db, 'reference, frequency, Jl, Ju, shape, fixShape, charge', 'Lines',
                                   [['lineVar'], [lineVar]], caller_name=__name__)[0]
        if not data:
            print("No such line: " + lineVar)
        else:
            self.name = iso + isovar
            self.isovar = isovar
            self.lineVar = lineVar
            self.ref = data[0]
            self.freq = data[1]
            self.Jl = data[2]
            self.Ju = data[3]
            self.shape = eval(data[4])
            self.fixShape = eval(data[5])
            elmass = data[6] * Physics.me_u
            print('loaded :', self.name)

        data = TiTS.select_from_db(db,
            'mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat, fixedBrat, intScale, fixedInt, relInt, m', 'Isotopes',
                                   [['iso'], [iso + isovar]], caller_name=__name__)[0]
        if not data:
            print("No such isotope: " + iso + isovar)
        else:
            self.mass = data[0] - elmass
            self.mass_d = data[1]
            self.I = data[2]
            self.center = data[3]
            self.Al = data[4]
            self.Bl = data[5]
            self.Au = data[6]
            self.Bu = data[7]
            self.fixArat = data[8]
            self.fixBrat = data[9]
            self.intScale = data[10]
            self.fixInt = data[11]
            if data[12]:
                self.relInt = eval(data[12])
            else:
                self.relInt = []
            if not data[13]:
                self.m = None
            else:
                self.m = DBIsotope(db, data[13], lineVar)