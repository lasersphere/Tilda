'''
Created on 15.05.2014

@author: hammen
'''

import os

from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.TLDImporter import TLDImporter
from Measurement.SimpleImporter import SimpleImporter


def load(file, db):
    e = os.path.splitext(file)[1]
    
    if e == '.txt':
        return KepcoImporterTLD(file, db)
    elif e == '.tld':
        return TLDImporter(file, db)
        
    else:
        print("Measload: Unknown file ending ", file)
        return None
