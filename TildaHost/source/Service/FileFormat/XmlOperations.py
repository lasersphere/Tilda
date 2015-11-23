"""

Created on '26.10.2015'

@author:'simkaufm'

"""
from datetime import datetime as dt
from lxml import etree as ET
from Service.VoltageConversions.VoltageConversions import get_voltage_from_18bit


def xmlFindOrCreateSubElement(parentEle, tagString, value=''):
    """
    finds or creates a Subelement with the tag tagString and text=value to the parent Element.
    :return: returns the SubElement
    """
    subEle = parentEle.find(tagString)
    if subEle == None:
        ET.SubElement(parentEle, tagString)
        return xmlFindOrCreateSubElement(parentEle, tagString, value)
    if value != '':
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
    isotopedict.update(isotopeStartTime=time)
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


def xmlAddCompleteTrack(rootEle, scanDict, data, track_name):
    """
    Add a complete Track to an lxml root element
    """
    datatype = scanDict['isotopeData']['type']
    pipeInternalsDict = scanDict['pipeInternals']
    nOfTrack = int(track_name[5:])
    trackDict = scanDict[track_name]
    trackDict.update(dacStartVoltage=get_voltage_from_18bit(trackDict['dacStartRegister18Bit']))
    trackDict.update(dacStepsizeVoltage=get_voltage_from_18bit(trackDict['dacStepSize18Bit'] + int(2 ** 17)))
    xmlWriteTrackDictToHeader(rootEle, nOfTrack, trackDict)
    xmlWriteToTrack(rootEle, nOfTrack, 'scalerArray', data, 'data')
    return rootEle


def xmlGetDataFromTrack(rootEle, nOfTrack, dataType):
    """
    Get Data From Track
    :param rootEle:  lxml.etree.Element, root of the xml tree
    :param nOfTrack: int, which Track should be written to
    :param dataType: str, valid: 'setOffset, 'measuredOffset', 'dwellTime10ns', 'nOfmeasuredSteps',
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
        return None


def xmlGetDictFromEle(element):
    """
    Converts an lxml Element into a python dictionary
    """
    return element.tag, dict(map(xmlGetDictFromEle, element)) or element.text