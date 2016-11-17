"""

Created on '26.10.2015'

@author:'simkaufm'

"""
from datetime import datetime as dt

from lxml import etree as ET

import Application.Config as Cfg


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
        if isinstance(val, dict):
            xmlWriteDict(xmlFindOrCreateSubElement(parentEle, key), val)
        else:
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
    xmlFindOrCreateSubElement(root, 'measureVoltPars')
    xmlFindOrCreateSubElement(root, 'tracks')
    return root


def xml_add_meas_volt_pars(meas_volt_pars_dict, root_element):
    """
    this will add the voltage measurement parameters to the measVoltPars SubElement
    :param meas_volt_pars_dict: dict, as in draftScanParameters.py
    :return: root_element
    """
    meas_volt_pars = xmlFindOrCreateSubElement(root_element, 'measureVoltPars')
    xmlWriteDict(meas_volt_pars, meas_volt_pars_dict)
    return root_element


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


def xmlAddCompleteTrack(rootEle, scanDict, data, track_name, datatype='scalerArray', parent_ele_str='data'):
    """
    Add a complete Track to an lxml root element
    :param rootEle: lxml.etree.Element, Element of loaded File
    :param scanDict: dict, dictionary containing all scan parameters
    :param data: array of data containing all scalers fpr this track
    :param track_name: str, name of track
    :param datatype: str, name of data that will be written to the parent_ele_str
    :param parent_ele_str: str, name of the subelement taht will be created/found in the selected track
    :return: rootEle
    """
    # datatype = scanDict['isotopeData']['type']
    # pipeInternalsDict = scanDict['pipeInternals']
    nOfTrack = int(track_name[5:])
    trackDict = scanDict[track_name]
    # this should be already included before:
    # trackDict.update(dacStartVoltage=get_voltage_from_18bit(trackDict['dacStartRegister18Bit']))
    # trackDict.update(dacStepsizeVoltage=VCon.get_stepsize_in_volt_from_18bit(trackDict['dacStepSize18Bit']))
    # trackDict.update(dacStopVoltage=get_voltage_from_18bit(
    #     VCon.calc_dac_stop_18bit(trackDict['dacStartRegister18Bit'],
    #                              trackDict['dacStepSize18Bit'],
    #                              trackDict['nOfSteps'])))
    xmlWriteTrackDictToHeader(rootEle, nOfTrack, trackDict)
    xmlWriteToTrack(rootEle, nOfTrack, datatype, data, parent_ele_str)
    return rootEle


def xml_create_autostart_root():
    root = ET.Element('Tilda_autostart_file_%s' % Cfg.version.replace('.', '_'))
    xmlFindOrCreateSubElement(root, 'workingDir', 'somepath')
    devs = xmlFindOrCreateSubElement(root, 'autostartDevices')
    xmlFindOrCreateSubElement(devs, 'dmms', '{\'dmm_name\':\'address\'}')
    xmlFindOrCreateSubElement(devs, 'powersupplies', '{\'powersup_name\':\'address\'}')
    return root


def xml_create_fpga_cfg_root():
    root = ET.Element('Tilda_fpga_cfg_file')
    fpgas = xmlFindOrCreateSubElement(root, 'fpgas')
    data_acq_fpga = xmlFindOrCreateSubElement(fpgas, 'data_acquisition_fpga')
    xmlFindOrCreateSubElement(data_acq_fpga, 'fpga_resource', 'Rio1')
    xmlFindOrCreateSubElement(data_acq_fpga, 'fpga_type', 'PXI-7852R')
    control_fpga = xmlFindOrCreateSubElement(fpgas, 'control_fpga')
    xmlFindOrCreateSubElement(control_fpga, 'fpga_resource', 'Rio0')
    xmlFindOrCreateSubElement(control_fpga, 'fpga_type', 'PXI-7852R')
    return root