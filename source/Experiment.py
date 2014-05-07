'''
Created on 23.03.2014

@author: hammen
'''

#import time

def getAccVolt(time = 0):
    '''Return the ion source voltage in V'''
    return 29956.21

def getLaserFreq(time = 0):
    """Return the laser frequency in MHz"""
    return 1398640292.89

def dirColTrue(time = 0):
    '''Return True for collinear, False for anticollinear laser configuration'''
    return True

def getVoltDivRatio(time = 0):
    '''Return the voltage divider ratio'''
    return 1000.

def lineToScan(lineV):
    '''Convert line voltage to scan voltage'''
    return lineV * getLineMult() + getLineOffset()

def getLineMult(time = 0):
    '''Kepco-Factor, should only be called by lineToScan'''
    return 50.

def getLineOffset(time = 0):
    '''Kepco-Offset, should only be calles by lineToScan'''
    return 0.