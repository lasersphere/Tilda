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

def xmlCreatBlankIsotope(isotopeData):
    """
    Builds the lxml Element Body for one isotope.
    Constant information for all isotopes is included in header.
    All Tracks are included in tracks.
    :param isotopeData: dict, containing: version, type, datetime, isotope, nOfTracks,
    colDirTrue, accVolt, laserFreq
    :return: lxml.etree.Element
    """
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

    for i in range(int(isotopeData['nOfTracks'])):
        xmlAddBlankTrack(tracks, i)
    return root

def xmlFillIsotopeData(rootEle, dataType, newData):
    """
    writes newData to dataType in the header of a rootEle.
    """
    head = rootEle.find('header')
    head.find(dataType).text = newData


def xmlAddBlankTrack(parent, tracknumber):
    """
    Adds a track to a parent
    """
    ET.SubElement(parent, 'track' + str(tracknumber))
    ET.SubElement(parent[tracknumber], 'setOffset')
    ET.SubElement(parent[tracknumber], 'measuredOffset')
    ET.SubElement(parent[tracknumber], 'dwellTime')
    ET.SubElement(parent[tracknumber], 'nOfmeasuredSteps')
    ET.SubElement(parent[tracknumber], 'nOfcompletededLoops')
    ET.SubElement(parent[tracknumber], 'voltArray')
    ET.SubElement(parent[tracknumber], 'timeArray')
    ET.SubElement(parent[tracknumber], 'scalerArray')


def xmlAddDataToTrack(rootEle, nOfTrack, dataType, newData):
    """
    replaces Data in nOfTrack from rootEle with newData of type dataType.
    :param rootEle: lxml.etree.Element, root of the xml tree
    :param nOfTrack: int, which Track should be written to
    :param dataType: str, valid: 'voltArray', 'timeArray', 'scalerArray'
    :param newData: np.array, newData that will replace all before.
    :return: None,
    """
    tracks = rootEle.find('tracks')
    try:
        #check if track is already created
        track = tracks.getchildren()[nOfTrack]
        track.find(dataType).text = repr(newData)
    except IndexError:
        #add track at next possible position.
        nOfTrack = len(tracks.getchildren())
        xmlAddDataToTrack(tracks, nOfTrack)
        track = tracks.getchildren()[nOfTrack]
        track.find(dataType).text = repr(newData)
