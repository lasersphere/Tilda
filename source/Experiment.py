'''
Created on 23.03.2014

@author: hammen
'''

#import time

def getAccVolt(time = 0):
    return 29956.21

def getLaserFreq(time = 0):
    """Return the laser frequency in MHz"""
    return 1398640292.89

def dirColTrue(time = 0):
    return True

def getVoltDivRatio(time = 0):
    return 1000.

def lineToScan(lineV):
    return lineV * getLineMult() + getLineOffset()

def getLineMult(time = 0):
    return 50.

def getLineOffset(time = 0):
    return 0.