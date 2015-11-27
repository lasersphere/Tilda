'''
Created on 15.05.2014

@author: hammen
'''

import os

from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.TLDImporter import TLDImporter

from Measurement.XMLImporter import XMLImporter
from Measurement.MCPImporter import MCPImporter
from Measurement.KepcoImporterMCP import KepcoImporterMCP
from Measurement.SimpleImporter import SimpleImporter



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

    elif e == '.mcp':
        f = MCPImporter(file)
        if not raw:
            f.preProc(db)
        return f

    elif e == '.kmcp':
        f = KepcoImporterMCP(file)
        if not raw:
            f.preProc(db)
        return f

    elif e == '.xml':
        file = file.replace('\\', '/')
        absfile = (os.path.join(os.getcwd(), file))
        f = XMLImporter(absfile)
        if not raw:
            f.preProc(db)
        return f
    else:
        return None

    
def check(end):
    return end in ['.txt', '.tld', '.xml', '.mcp', '.kmcp']
