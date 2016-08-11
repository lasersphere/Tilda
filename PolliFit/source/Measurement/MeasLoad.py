'''
Created on 15.05.2014

@author: hammen
'''

import os

from Measurement.ALIVEImporter import ALIVEImporter
from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.MCPImporter import MCPImporter
from Measurement.TLDImporter import TLDImporter
from Measurement.XMLImporter import XMLImporter


def load(file, db, raw=False, x_as_voltage=True, softw_gates=None):
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

    elif e == '.mcp':
        f = MCPImporter(file)
        if not raw:
            f.preProc(db)
        return f

    elif e == '.xml':
        f = XMLImporter(file, x_as_voltage, softw_gates=softw_gates)
        if not raw:
            f.preProc(db)
        return f

    elif e == '.dat':
        f = ALIVEImporter(file)
        if not raw:
            f.preProc(db)
        return f
    else:
        return None
    
def check(end):
    return end in ['.txt', '.tld', '.xml', '.mcp', '.dat']
