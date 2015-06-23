'''
Created on 21.01.2015

@author: skaufmann
'''

from datetime import datetime as dt
import lxml.etree as ET
import numpy as np


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
    voltage = (voltage >> 2) & ((2 ** 18) - 1) #shift by 2 and delete higher parts of payload
    index = np.where(voltArray == voltage)
    if len(index[0]) == 0:
        #voltage not yet in array, put it at next empty position
        index = np.where(voltArray == 0)[0][0]
    else:
        #voltage already in list, take the found index
        index = index[0][0]
    np.put(voltArray, index, voltage)
    return (index, voltArray)

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

def xmlCreateIsotope(isotopeDict):
    """
    Builds the lxml Element Body for one isotope.
    Constant information for all isotopes is included in header.
    All Tracks are included in tracks.
    :param isotopeDict: dict, containing: see OneNote
    :return: lxml.etree.Element
    """
    root = ET.Element('TrigaLaserData')

    header = ET.SubElement(root, 'header')
    xmlWriteIsoDictToHeader(header, isotopeDict)

    tracks = ET.SubElement(root, 'tracks')

    for i in range(int(isotopeDict['nOfTracks'])):
        xmlAddBlankTrack(tracks, i)
    return root


def xmlWriteIsoDictToHeader(header, isotopedict):
    """
    write the complete isotopedict to the header of the xml structure
    """
    header.find('datetime').text = str(dt.now())
    for key, val in isotopedict.items():
        try:
            header.find(key).text = val
        except KeyError:
            ET.SubElement(header, key, val)

def xmlAddBlankTrack(parent, tracknumber):
    """
    Adds a track to a parent
    """
    ET.SubElement(parent, 'track' + str(tracknumber))
    ET.SubElement(parent[tracknumber], 'header')
    ET.SubElement(parent[tracknumber], 'data')

def xmlWriteToTrack(rootEle, nOfTrack, dataType, newData, headOrDatStr='header',):
    """
    replaces Data in nOfTrack from rootEle with newData of type dataType.
    :param rootEle: lxml.etree.Element, root of the xml tree
    :param nOfTrack: int, which Track should be written to
    :param dataType: str,
    :param newData: np.array, newData that will replace all before.
    """
    tracks = rootEle.find('tracks')
    try: #check if track and dataType is already created
        track = tracks.getchildren()[nOfTrack]
        headOrData = track.find(headOrDatStr)
        headOrData.find(dataType).text = repr(newData)
    except IndexError:  #track not found, at next possible position.
        xmlAddBlankTrack(tracks, nOfTrack)
        xmlWriteToTrack(rootEle, nOfTrack, dataType, newData)
    except KeyError: #Element not found, create
        ET.SubElement(headOrData, dataType, newData)


def xmlAddCompleteTrack(rootEle, scanDict, data):
    # data must be tuple of form: (self.voltArray, self.timeArray, self.scalerArray)
    pipeInternalsDict = scanDict['pipeInternals']
    nOfTrack = pipeInternalsDict['activeTrackNumber']
    trackDict = scanDict['activeTrackPar']
    for key, val in trackDict.items(): #write header
        xmlWriteToTrack(rootEle, nOfTrack, key, val)
    dataFormat = ('voltArray', 'timeArray', 'scalerArray')
    for i in dataFormat:
        xmlWriteToTrack(rootEle, nOfTrack, dataFormat[i], data[i], 'data')



def xmlGetDataFromTrack(rootEle, nOfTrack, dataType):
    """

    :param rootEle:  lxml.etree.Element, root of the xml tree
    :param nOfTrack: int, which Track should be written to
    :param dataType: str, valid: 'setOffset, 'measuredOffset', 'dwellTime', 'nOfmeasuredSteps',
     'nOfclompetedLoops', 'voltArray', 'timeArray', 'scalerArray'
    :param returnType: int or tuple of int, shape of the numpy array, 0 if output in textfrom is desired
    :return: Text
    """
    try:
        actTrack = rootEle.find('tracks').find('track' + str(nOfTrack))
        dataText = actTrack.find(str(dataType)).text
        return dataText
    except:
        print('error while searching ' +str(dataType) +  ' in track' + str(nOfTrack) + ' in ' + str(rootEle))


def numpyArrayFromString(string, shape):
    """
    converts a text array saved in an lxml.etree.Element
    using the function xmlWriteToTrack back into a numpy array
    :param string: str, array
    :param shape: int, or tuple of int, the shape of the output array
    :return: numpy array containing the desired values
    """
    string = string.replace('\\n', '').replace('[', '').replace(']', '').replace('  ', ' ')
    result = np.fromstring(string[1:-1], dtype=np.uint32, sep=' ')
    result = result.reshape(shape)
    return result

