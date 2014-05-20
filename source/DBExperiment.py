'''
Created on 19.05.2014

@author: hammen
'''
import sqlite3


class DBExperiment(object):
    '''
    A sqlite database driven version fo the isotope object
    '''

    def __init__(self, file):

        self.con = sqlite3.connect(file)
        self.cur = self.con.cursor()

        
    def __del__(self):
        self.cur.close()
        self.con.close()
        
    def getAccVolt(self, time = 0):
        '''Return the ion source voltage in V'''
    #    return 29956.21
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'accVolt' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return 9995.5
    
    def getLaserFreq(self, time = 0):
        """Return the laser frequency in MHz"""
    #    return 1398640292.89
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'laserFreq' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return Physics.freqFromWavenumber(12586.3*2)
        
    
    def dirColTrue(self, time = 0):
        '''Return True for collinear, False for anticollinear laser configuration'''
    #    return True
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'dirColTrue' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return False
    
    
    def getVoltDivRatio(self, time = 0):
        '''Return the voltage divider ratio'''
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'voltDivRatio' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return 999.985
    
    def lineToScan(self, lineV, time = 0):
        '''Convert line voltage to scan voltage'''
        return lineV * self.getLineMult(time) + self.getLineOffset(time)
    
    def getLineMult(self, time = 0):
        '''Kepco-Factor, should only be called by lineToScan'''
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'lineMult' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return 50.038915763
    
    def getLineOffset(self, time = 0):
        '''Kepco-Offset, should only be called by lineToScan'''
        self.cur.execute('''SELECT value FROM Experiment WHERE name = 'lineOffset' AND ? BETWEEN begin AND end''', (time,))
        return self.cur.fetchall()[0][0]
        #return 0.0142577160157
        
def createDB(path = 'exp.sqlite'):
    '''Create a sqlite database with the appropriate structure'''
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS Experiment (
    begin DATETIME,
    end DATETIME,
    name TEXT,
    value FLOAT
    )''')
    
    con.commit()
    con.close()
    