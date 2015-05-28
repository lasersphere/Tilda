'''
Created on 21.01.2015

@author: skaufmann
'''

from datetime import datetime as dt
import lxml.etree as ET
import numpy as np

counter = 0

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

def findVoltage(voltage, voltArray):
    """
    find the index of voltage in voltArray. If not existant, create.
    :return: (int, np.array), index and VoltageArray
    """
    '''payload is 23-Bits, Bits 2 to 20 is the DAC register'''
    voltage = (voltage >> 2) & ((2 ** 18) - 1)
    index = np.where(voltArray == voltage)
    if len(index[0]) == 0:
        #voltage not yet in array, put it at next empty position
        index = np.where(voltArray == 0)[0][0]
    else:
        #voltage already in list, take the found index
        index = index[0][0]
    np.put(voltArray, index, voltage)
    print('yep its a voltage: ' + str(index))
    return (index, voltArray)

def mcsSum(element, actVoltInd, sumArray):
    """
    Add new Scaler event on previous acquired ones. Treat each scaler seperatly.
    :return: np.array, sum
    """
    timestamp = element['payload']
    activePmts = (element['firstHeader'] << 4) + element['secondHeader'] #glue header back together
    for i in range(8):
        if activePmts & (2 ** i):
            sumArray[actVoltInd, timestamp, i] += 1 #timestamp equals index in timeArray
    return sumArray

def xmlFormat(isotopeData):
    root = ET.Element('TrigaLaserData')

    header = ET.SubElement(root, 'header')

    version = ET.SubElement(header, 'version')
    version.text = isotopeData['version']

    typ = ET.SubElement(header, 'type')
    typ.text = isotopeData['type']

    datetime = ET.SubElement(header, 'datetime')
    datetime.text = str(dt.now())

    iso = ET.SubElement(header, 'isotope')
    iso.text = isotopeData['isotope']

    nrTracks = ET.SubElement(header, 'nrTracks')
    nrTracks.text = isotopeData['nOfTracks']

    direc = ET.SubElement(header, 'colDirTrue')
    direc.text = isotopeData['colDirTrue']

    accVolt = ET.SubElement(header, 'accVolt')
    accVolt.text = isotopeData['accVolt']

    laserFreq = ET.SubElement(header, 'laserFreq')
    laserFreq.text = isotopeData['laserFreq']


    tracks = ET.SubElement(root, 'tracks')

    track0 = ET.SubElement(tracks, 'track0')

    leftLine0 = ET.SubElement(track0, 'leftLine')
    leftLine0.text = '-200'

    data0 = ET.SubElement(track0, 'data')
    data0.text = repr(isotopeData['scalerArray'])


    tree = ET.ElementTree(root)
    tree.write(isotopeData['saveToFile'], pretty_print = True)
