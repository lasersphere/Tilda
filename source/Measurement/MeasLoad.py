'''
Created on 15.05.2014

@author: hammen
'''

import os

from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.TLDImporter import TLDImporter
#from Measurement.SimpleImporter import SimpleImporter


def load(file, db, raw = False):
    e = os.path.splitext(file)[1]
    
    if e == '.txt':
        f = KepcoImporterTLD(file)
        if not raw:
            f.preProc(db)
        return f
    
    elif e == '.tld':
        f = TLDImporter(file)
        if not raw:
            f.preProc(db)
        return f
    
    else:
        return None

    
def check(end):
    return end in ['.txt', '.tld']
