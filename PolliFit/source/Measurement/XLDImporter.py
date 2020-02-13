'''
Created on 30.04.2014

@author: hammen
'''
import sqlite3
from datetime import datetime
import os

import numpy as np
import lxml.etree as ET

import TildaTools

from Measurement.SpecData import SpecData

class XLDImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    
     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("TLDImporter is reading file", path)
        super(XLDImporter, self).__init__()

        
        self.file = os.path.basename(path)

        try:
            tree = ET.parse(path)
        except:
            raise
        
        header = tree.find('header')
        
        self.version = header.findtext('version')
        self.type = header.findtext('type')
        self.datetime = datetime(header.findtext('datetime'))
        self.nrTracks = header.findtext('nrTracks')
        self.nrScalers = header.findtext('nrScalers')
        self.col = eval(header.findtext('colDirTrue'))

 
    def preProc(self, db):
        print('Kepco importer is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.colDirTrue, self.line, self.type, self.voltDivRatio, self.lineMult, self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('TLDImporter: No DB-entry found!')
                
        # for i in range(len(self.x[0])):
        #     scanvolt = self.lineMult * (self.x[0][i]) + self.lineOffset + self.offset
        #     self.x[0][i] = self.accVolt - scanvolt
        self.x[0] = TildaTools.line_to_total_volt(self.x[0], self.lineMult, self.lineOffset, self.offset, self.accVolt,
                                                  {'offset': 1.0, 'accVolt': 1.0, 'lineMult': 1.0})
        
        con.close()
 
    
    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ? WHERE file = ?''', (self.date, self.file))     
        con.close()
    
