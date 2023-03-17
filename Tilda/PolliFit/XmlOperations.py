"""

Created on '26.10.2015'

@author:'simkaufm'

"""
from datetime import datetime as dt

import numpy as np
from lxml import etree as ET


def xmlFindOrCreateSubElement(parentEle, tagString, value=''):
    """
    finds or creates a Subelement with the tag tagString and text=value to the parent Element.
    Try not to use colons in the tagstring!
    :return: returns the SubElement
    """
    if ':' in tagString:  # this will otherwise cause problems with namespace in xml!
        tagString = tagString.replace(':', '.')
    subEle = parentEle.find(tagString)
    if subEle == None:
        ET.SubElement(parentEle, tagString)
        return xmlFindOrCreateSubElement(parentEle, tagString, value)
    # print(dt.now(), ' string conversion started ', tagString, type(value))
    if isinstance(value, np.ndarray):
        # print('numpy conversion started', value.dtype)
        val_str = ''
        if value.dtype == [('sc', '<u2'), ('step', '<u4'), ('time', '<u4'), ('cts', '<u4')]:
            np.savetxt('temp.out', value, fmt=['%d', '%d', '%d', '%d'])
            with open('../temp.out', 'r') as f:
                ret = f.readlines()
            for each in ret:
                each = each.replace(' ', ', ').replace('\n', '')
                each = '(' + each + ') '
                val_str += each
            val_str = '[' + val_str + ']'
        elif value.dtype == np.int32:
            np.savetxt('temp.out', value, fmt='%d')
            with open('../temp.out', 'r') as f:
                ret = f.readlines()
            for each in ret:
                val_str += '[' + each + ']'
            val_str = '[' + val_str + ']'
        else:
            val_str = str(value)
    else:
        # print('normal str conv')
        val_str = str(value)
    # print(dt.now(), ' string conversion done ', tagString)
    if val_str:
        subEle.text = val_str
        # print(dt.now(), ' subelement text is set.', tagString)
    return subEle


def xmlWriteDict(parentEle, dictionary, exclude=[]):
    """
    finds or creates a Subelement with the tag tagString and text=value to the
    parent Element for each key and value pair in the dictionary.
    :return: returns the modified parent Element.
    """
    for key, val in sorted(dictionary.items(), key=str):
        if key not in exclude:
            if isinstance(val, dict):
                xmlWriteDict(xmlFindOrCreateSubElement(parentEle, key), val)
            else:
                xmlFindOrCreateSubElement(parentEle, key, val)
    return parentEle


def xmlCreateIsotope(isotopeDict, take_time_now=True):
    """
    Builds the lxml Element Body for one isotope.
    Constant information for all isotopes is included in header.
    All Tracks are included in tracks.
    :param isotopeDict: dict, containing: see OneNote
    :return: lxml.etree.Element
    """
    root = ET.Element('TrigaLaserData')
    xmlWriteIsoDictToHeader(root, isotopeDict, take_time_now=take_time_now)
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


def xmlWriteIsoDictToHeader(rootEle, isotopedict, take_time_now=True):
    """
    write the complete isotopedict and the datetime to the header of the xml structure
    """
    head = xmlFindOrCreateSubElement(rootEle, 'header')
    if take_time_now:
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


def xmlAddCompleteTrack(rootEle, scanDict, data, track_name, datatype='scalerArray',
                        parent_ele_str='data', data_explanation_str=''):
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
    # seq_type = scanDict.get('isotopeData', {}).get('type', 'cs')
    # pipeInternalsDict = scanDict['pipeInternals']
    nOfTrack = int(track_name[5:])
    trackDict = scanDict[track_name]
    # write header
    xmlWriteTrackDictToHeader(rootEle, nOfTrack, trackDict)
    # write explanation of data
    if data_explanation_str == '':
        data_explanation_str = get_data_explanation_str(scanDict, datatype)
    if data_explanation_str:
        xmlWriteToTrack(rootEle, nOfTrack, datatype + '_explanation', data_explanation_str, parent_ele_str)
    # write the data
    xmlWriteToTrack(rootEle, nOfTrack, datatype, data, parent_ele_str)
    return rootEle


def get_data_explanation_str(scan_dict, datatype):
    """ use this to stroe an explanation of the dataformat within the same xml file. """
    seq_type = scan_dict.get('isotopeData', {}).get('type', 'cs')
    data_explanation = ''
    if seq_type in ['cs', 'csdummy']:  # not time resolved
        data_explanation = 'continously acquired data.' \
                           ' List of Lists, each list represents the counts of one' \
                           ' scaler as listed in activePmtList.' \
                           'Dimensions are: (len(activePmtList), nOfSteps), datatype: np.int32'
    elif seq_type in ['trs', 'trsdummy', 'tipa']:  # time resolved
        if 'voltage_projection' in datatype:
            data_explanation = datatype + ' of the time resolved data.' \
                                          ' List of Lists, each list represents the counts of one' \
                                          ' scaler as listed in activePmtList.' \
                                          'Dimensions are: (len(activePmtList), nOfSteps), datatype: np.int32'
        elif 'time_projection' in datatype:
            data_explanation = datatype + ' of the time resolved data.' \
                                          ' List of Lists, each list represents the counts of one' \
                                          ' scaler as listed in activePmtList.' \
                                          'Dimensions are: (len(activePmtList), nOfBins), datatype: np.int32'
        else:
            data_explanation = 'time resolved data. List of tuples, each tuple consists of: \n' \
                               '(scaler_number, line_voltage_step_number, time_stamp, number_of_counts),' \
                               ' datatype: np.int32'
    elif seq_type in ['kepco']:  # kepco scan
        data_explanation = 'kepco scan data. ' \
                           ' List of Lists, each list holds the readings of one' \
                           ' multimeter as listed in measureVoltPars.' \
                           'Dimensions are: (len(activePmtList), nOfSteps), datatype: np.float'
    return data_explanation


def xml_create_autostart_root(version):
    root = ET.Element('Tilda_autostart_file_%s' % version.replace('.', '_'))
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