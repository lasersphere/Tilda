'''
Created on 21.01.2015

@author: skaufmann
'''

from datetime import datetime as dt
import lxml.etree as ET
import numpy as np
import Service.dataFormat as dataForm


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

def xmlFindOrCreateSubElement(parentEle, tagString, value=None):
    """
    finds or creates a Subelement with the tag tagString and text=value to the parent Element.
    :return: returns the SubElement
    """
    subEle = parentEle.find(tagString)
    if subEle == None:
        ET.SubElement(parentEle, tagString)
        return xmlFindOrCreateSubElement(parentEle, tagString, value)
    if value != None:
        subEle.text = str(value)
    return subEle

def xmlWriteDict(parentEle, dictionary):
    """
    finds or creates a Subelement with the tag tagString and text=value to the
    parent Element for each key and value pair in the dictionary.
    :return: returns the modified parent Element.
    """
    for key, val in sorted(dictionary.items(), key=str):
        xmlFindOrCreateSubElement(parentEle, key, val)
    return parentEle

def xmlCreateIsotope(isotopeDict):
    """
    Builds the lxml Element Body for one isotope.
    Constant information for all isotopes is included in header.
    All Tracks are included in tracks.
    :param isotopeDict: dict, containing: see OneNote
    :return: lxml.etree.Element
    """
    root = ET.Element('TrigaLaserData')
    xmlWriteIsoDictToHeader(root, isotopeDict)
    xmlFindOrCreateSubElement(root, 'tracks')
    return root

def xmlWriteIsoDictToHeader(rootEle, isotopedict):
    """
    write the complete isotopedict and the datetime to the header of the xml structure
    """
    head = xmlFindOrCreateSubElement(rootEle, 'header')
    time = str(dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    isotopedict.update(datetime=time)
    xmlWriteDict(head, isotopedict)
    return rootEle

def xmlFindTrackInTracks(rootEle, nOfTrack):
    """
    Finds or Creates the Track with number nOfTrack -> track SubElement
    """
    tracks = xmlFindOrCreateSubElement(rootEle, 'tracks')
    track = xmlFindOrCreateSubElement(tracks, 'track' + str(nOfTrack))
    return track

def xmlWriteToTrack(rootEle, nOfTrack, dataType, newData, headOrDatStr='header',):
    """
    Writes newData to the Track with number nOfTrack. Either in the header or in the data SubElement
    -> Subelement that has been written to.
    """
    track = xmlFindTrackInTracks(rootEle, nOfTrack)
    workOn = xmlFindOrCreateSubElement(track, headOrDatStr)
    return xmlFindOrCreateSubElement(workOn, dataType, newData)

def xmlWriteTrackDictToHeader(rootEle, nOfTrack, trackdict):
    """
    write the whole trackdict to the header of the specified Track
    :return:rootEle
    """
    track = xmlFindTrackInTracks(rootEle, nOfTrack)
    headerEle = xmlFindOrCreateSubElement(track, 'header')
    xmlWriteDict(headerEle, trackdict)
    return rootEle

def xmlAddCompleteTrack(rootEle, scanDict, data):
    datatype = scanDict['isotopeData']['type']
    dataFormat = dataForm.dataformat[str(datatype)]
    pipeInternalsDict = scanDict['pipeInternals']
    nOfTrack = pipeInternalsDict['activeTrackNumber']
    trackDict = scanDict['activeTrackPar']
    xmlWriteTrackDictToHeader(rootEle, nOfTrack, trackDict)
    for i, j in enumerate(dataFormat):
        xmlWriteToTrack(rootEle, nOfTrack, j, data[i], 'data')
    return rootEle

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
        actTrack = rootEle.find('tracks').find('track' + str(nOfTrack)).find('data')
        dataText = actTrack.find(str(dataType)).text
        return dataText
    except:
        print('error while searching ' +str(dataType) +  ' in track' + str(nOfTrack) + ' in ' + str(rootEle))

def xmlGetDictFromEle(element):
    """
    Converts an lxml Element into a python dictionary
    """
    return element.tag, dict(map(xmlGetDictFromEle, element)) or element.text

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

