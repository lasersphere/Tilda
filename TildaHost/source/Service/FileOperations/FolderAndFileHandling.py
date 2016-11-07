"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import os
import pickle
from copy import deepcopy

import numpy as np

import Service.Scan.ScanDictionaryOperations as SdOp
import TildaTools as Tits
import Tools
from Service.FileOperations.XmlOperations import xmlCreateIsotope, xml_add_meas_volt_pars,\
    xmlAddCompleteTrack, xml_create_autostart_root, xmlWriteDict, xml_create_fpga_cfg_root
from TildaTools import save_xml


def findTildaFolder(path=os.path.dirname(os.path.abspath(__file__))):
    """
    tries to find the Tilda folder relative to execution file
    :return: str, path of Tilda Folder
    """
    if 'Tilda' in os.path.split(path)[0]:
        path = findTildaFolder(os.path.dirname(path))
    elif 'Tilda' in os.path.split(path)[1]:
        path = path
    else:
        path = 'could not find Tilda folder'
    return path


def nameFile(path, subdir, fileName, prefix='', suffix='.tld'):
    """
    find an unused valid filename.
    :return: str, path/subdir/timestamp_prefix_fileName + suffix
    """
    storagePath = os.path.join(path, subdir)
    if not os.path.exists(storagePath):
        os.makedirs(storagePath)
    filepath = os.path.join(storagePath, prefix + '_' + fileName)
    i = 0
    file = filepath + '_' + str('{0:03d}'.format(i)) + suffix
    if not os.path.isfile(file):
        return filepath + '_' + str('{0:03d}'.format(i)) + suffix
    while os.path.isfile(file):
        i += 1
        file = filepath + '_' + str('{0:03d}'.format(i)) + suffix
    return file


def createXmlFileOneIsotope(scanDict, seq_type=None, filename=None):
    """
    creates an .xml file for one Isotope. Using the Filestructure as stated in OneNote.
    :param scanDict: {'isotopeData', 'track0', 'pipeInternals'}
    :return:str, filename
    """
    isodict = deepcopy(scanDict['isotopeData'])
    meas_volt_dict = deepcopy(scanDict['measureVoltPars'])
    if seq_type is not None:
        isodict['type'] = seq_type
    root = xmlCreateIsotope(isodict)
    xml_add_meas_volt_pars(meas_volt_dict, root)
    if filename is None:
        path = scanDict['pipeInternals']['workingDirectory']
        filename = nameFileXml(isodict, path)
    print('creating .xml File: ' + filename)
    save_xml(root, filename, False)
    # now add it to the database:
    db_name = os.path.basename(scanDict['pipeInternals']['workingDirectory']) + '.sqlite'
    db = scanDict['pipeInternals']['workingDirectory'] + '\\' + db_name
    if os.path.isfile(db):
        os.chdir(scanDict['pipeInternals']['workingDirectory'])
        relative_filename = os.path.normpath(
            os.path.join(os.path.split(os.path.dirname(filename))[1], os.path.basename(filename)))
        Tools._insertFile(relative_filename, db)
    return filename


def nameFileXml(isodict, path):
    """
    finds a filename for the xml file in subdirectory 'sums'
    :param scanDict: {'isotopeData', 'track0', 'pipeInternals'}
    :return:str, filename
    """
    # path = scanDict['pipeInternals']['workingDirectory']
    nIso = isodict['isotope']
    seq_type = isodict['type']
    filename = nameFile(path, 'sums', seq_type, nIso, '.xml')
    return filename


def savePickle(data, pipeDataDict, ending='.raw'):
    """
    saves data using the pickle module
    """
    path = pipeDataDict['pipeInternals']['workingDirectory']
    xml_path = os.path.split(pipeDataDict['pipeInternals']['activeXmlFilePath'])[1].split('.')[0]
    path = nameFile(path, 'raw',
                    xml_path,
                    '',
                    ending)
    save_pickle_simple(path, data)
    return path


def loadPickle(file):
    """
    loads the content of a binary file using the pickle module.
    """
    stream = open(file, 'rb')
    data = pickle.load(stream)
    stream.close()
    return data


def save_pickle_simple(path, data):
    """
    simple function to write some pickleable data to a path
    """
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()
    return path


def saveRawData(data, pipeData, nOfSaves):
    """
    function to save the raw data using pickle.
    :return nOfSaves
    """
    if np.count_nonzero(data) > 0:
        savedto = savePickle(data, pipeData)
        nOfSaves += 1
        logging.info('saving raw data to: ' + str(savedto))
    return nOfSaves


def savePipeData(pipeData, nOfSaves):
    savedto = savePickle(pipeData, pipeData, '.pipedat')
    logging.info('saving pipe data to: ' + str(savedto))
    nOfSaves += 1
    return nOfSaves


def save_spec_data(spec_data, scan_dict):
    """
    this will write the necessary values of the spec_data to an already existing xml file
    :param scan_dict: dict, containing all scan informations
    :param spec_data: spec_data, as a result from XmlImporter()
    :return: 
    """
    try:
        try:
            time_res = len(spec_data.time_res) # if there are any values in here, it is a time resolved measurement
        except Exception as e:
            time_res = False
        existing_xml_fil_path = scan_dict['pipeInternals']['activeXmlFilePath']
        root_ele = Tits.load_xml(existing_xml_fil_path)
        track_nums, track_num_lis = SdOp.get_number_of_tracks_in_scan_dict(scan_dict)
        for track_ind, tr_num in enumerate(track_num_lis):
            track_name = 'track' + str(tr_num)
            if time_res:
                scan_dict[track_name]['softwGates'] = spec_data.softw_gates[track_ind]
                xmlAddCompleteTrack(root_ele, scan_dict, spec_data.time_res_zf[track_ind], track_name)
                xmlAddCompleteTrack(
                    root_ele, scan_dict, spec_data.cts[track_ind], track_name, datatype='voltage_projection',
                    parent_ele_str='projections')
                xmlAddCompleteTrack(
                    root_ele, scan_dict, spec_data.t_proj[track_ind], track_name, datatype='time_projection',
                    parent_ele_str='projections')
            else:  # not time resolved
                scan_dict[track_name]['softwGates'] = []
                xmlAddCompleteTrack(root_ele, scan_dict, spec_data.cts[track_ind], track_name)
        Tits.save_xml(root_ele, existing_xml_fil_path, False)
    except Exception as e:
        print('error while saving: ', e)


def write_to_auto_start_xml_file(autostart_dict=None):
    """
    will create an editable autostart file in Tildahost/source/autostart.xml
    will write autostart_dict to this file if not None, else this will be untouched.
    :return: path of autostart file
    """
    main_path = os.path.join(findTildaFolder(), 'TildaHost', 'source', 'autostart.xml')
    if os.path.isfile(main_path):
        root, root_dict = load_auto_start_xml_file(main_path)
    else:
        root = xml_create_autostart_root()
    if autostart_dict is not None:
        xmlWriteDict(root, autostart_dict)
    Tits.save_xml(root, main_path)
    return main_path


def load_auto_start_xml_file(path):
    if os.path.isfile(path):
        root_ele = Tits.load_xml(path)
        root_dict = Tits.xml_get_dict_from_ele(root_ele)
        return root_ele, root_dict
    else:
        return None


def load_fpga_xml_config_file():
    print('loading fpga cfg')
    path = os.path.join(findTildaFolder(), 'TildaHost', 'source',
                             'Driver', 'DataAcquisitionFpga', 'fpga_config.xml')
    if os.path.isfile(path):
        root_ele = Tits.load_xml(path)
        root_dict = Tits.xml_get_dict_from_ele(root_ele)[1]
        return root_ele, root_dict
    else:
        root = xml_create_fpga_cfg_root()
        Tits.save_xml(root, path)
        root_ele = Tits.load_xml(path)
        root_dict = Tits.xml_get_dict_from_ele(root_ele)[1]
        return root_ele, root_dict
