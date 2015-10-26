"""
Created on 21.01.2015

@author: skaufmann
"""

from datetime import datetime as dt
import numpy as np
import copy
import ast


def split32bData(int32bData):
    """
    seperate header, headerIndex and payload from each other
    :param int32bData:
    :return: tuple, (firstHeader, secondHeader, headerIndex, payload)
    """
    headerLength = 8
    firstHeader = int32bData >> (32 - int(headerLength/2))
    secondHeader = int32bData >> (32 - headerLength) & ((2 ** 4) - 1)
    headerIndex = (int32bData >> (32 - headerLength - 1)) & 1
    payload = int32bData & ((2 ** 23) - 1)
    return (firstHeader, secondHeader, headerIndex, payload)


def trsSum(element, actVoltInd, sumArray, activePmtList=range(8)):
    """
    Add new Scaler event on previous acquired ones. Treat each scaler seperatly.
    :return: np.array, sum
    """
    timestamp = element['payload']
    pmtsWithEvent = (element['firstHeader'] << 4) + element['secondHeader'] #glue header back together
    for ind, val in enumerate(activePmtList):
        if pmtsWithEvent & (2 ** val):
            sumArray[actVoltInd, timestamp, ind] += 1 #timestamp equals index in timeArray
    return sumArray


def numpyArrayFromString(string, shape):
    """
    converts a text array saved in an lxml.etree.Element
    using the function xmlWriteToTrack back into a numpy array
    :param string: str, array
    :param shape: int, or tuple of int, the shape of the output array
    :return: numpy array containing the desired values
    """
    string = string.replace('\\n', '').replace('[', '').replace(']', '').replace('  ', ' ')
    result = np.fromstring(string, dtype=np.uint32, sep=' ')
    result = result.reshape(shape)
    return result


def eval_str_vals_in_dict(dicti):
    """
    function to convert the values of a dictionary to int, float or list, if it is possible
    """
    for key, val in dicti.items():
        dicti[key] = ast.literal_eval(val)
    return dicti


def addWorkingTimeToTrackDict(trackDict):
    """adds the timestamp to the working time of the track"""
    time = str(dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    if 'workingTime' in trackDict:
        if trackDict['workingTime'] == None:
            worktime = []
        else:
            worktime = trackDict['workingTime']
    else:
        worktime = []
    worktime.append(time)
    trackDict.update(workingTime=worktime)
    return trackDict


def convertScanDictV104toV106(scandict, draftScanDict):
    """
    converts a scandictionary created in Version 1.04 to the new format as it should be in v1.06
    was needed for working with the collected .raw data from 29.07.2015.
    """
    trackdft = draftScanDict['activeTrackPar']
    track = scandict['activeTrackPar']
    trackrenamelist = [('start', 'dacStartRegister18Bit'),
                       ('stepSize', 'dacStepSize18Bit'),
                       ('heinzingerOffsetVolt', 'postAccOffsetVolt'),
                       ('heinzingerControl', 'postAccOffsetVoltControl'),
                       ('dwellTime', 'dwellTime10ns')]
    track['workingTime'] = ['unknown']
    track['colDirTrue'] = scandict['isotopeData']['colDirTrue']
    scandict['isotopeData']['isotopeStartTime'] = scandict['isotopeData']['datetime']
    scandict['measureVoltPars'] = {k: v for (k, v) in track.items() if k in ['measVoltTimeout10ns', 'measVoltPulseLength25ns']}

    scandict['isotopeData'].pop('colDirTrue')
    scandict['isotopeData'].pop('datetime')
    [track.pop(k) for k in ['measVoltTimeout10ns', 'measVoltPulseLength25ns', 'VoltOrScaler', 'measureOffset']]
    for oldkey, newkey in trackrenamelist:
        track[newkey] = track.pop(oldkey)
    scandict['isotopeData']['version'] = 1.06
    return scandict


def createXAxisFromTrackDict(trackd):
    """
    uses a track dictionary to create the x axis, starting with dacStartRegister18Bit,
    length is nOfSteps and stepsize is dacStepSize18Bit
    """
    dacStart18Bit = trackd['dacStartRegister18Bit']
    dacStepSize18Bit = trackd['dacStepSize18Bit']
    nOfsteps = trackd['nOfSteps']
    dacStop18Bit = dacStart18Bit + (dacStepSize18Bit * nOfsteps)
    x = np.arange(dacStart18Bit, dacStop18Bit, dacStepSize18Bit)
    return x


def createDefaultScalerArrayFromScanDict(scand, dft_val=0):
    """
    create empty ScalerArray, size is determined by the activeTrackPar in the scan dictionary
    """
    trackd = scand['activeTrackPar']
    nOfSteps = trackd['nOfSteps']
    nofScaler = len(trackd['activePmtList'])
    arr = np.full((nOfSteps, nofScaler), dft_val, dtype=np.uint32)
    return arr

def createDefaultVoltArrayFromScanDict(scand, dft_val=(2 ** 30)):
    """
    create Default Voltage array, with default values in dft_val
    (2 ** 30) is chosen, because this is an default value which is not reachable by the DAC
    """
    trackd = scand['activeTrackPar']
    nOfSteps = trackd['nOfSteps']
    arr = np.full((nOfSteps,), dft_val, dtype=np.uint32)

    return arr