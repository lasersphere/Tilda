'''
Created on 15.05.2014

@author: hammen
'''

import os

from Tilda.PolliFit.Measurement.ALIVEImporter import ALIVEImporter
from Tilda.PolliFit.Measurement.KepcoImporterTLD import KepcoImporterTLD
from Tilda.PolliFit.Measurement.MCPImporter import MCPImporter
from Tilda.PolliFit.Measurement.TLDImporter import TLDImporter
from Tilda.PolliFit.Measurement.XMLImporter import XMLImporter
from Tilda.PolliFit.Measurement.BeaImporter import BeaImporter
from Tilda.PolliFit.Measurement.SimpleImporter import SimpleImporter


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

    elif e == '.sp':
        f = SimpleImporter(file)
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

    elif e == '.bea':
        f = BeaImporter(file,0,761360000,True)
        if not raw:
            f.preProc(db)
        return f
    else:
        return None


def check(end):
    return end in ['.txt', '.tld', '.xml', '.mcp', '.dat', '.bea', '.sp']
