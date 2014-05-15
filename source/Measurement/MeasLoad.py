'''
Created on 15.05.2014

@author: hammen
'''

import os

from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.SimpleImporter import SimpleImporter


def load(file):
    e = os.path.splitext(file)[1]
    
    if e == '.txt':
        return KepcoImporterTLD(file)
    elif e == '.tld':
        pass
        
    else:
        raise Exception("MeasLoad: Unknown file ending " + e + " in " + file)
