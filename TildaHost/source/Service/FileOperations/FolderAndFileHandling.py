"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import os
import pickle

import numpy as np

import TildaTools as Tits
from TildaTools import nameFile
from XmlOperations import xml_create_autostart_root, xmlWriteDict, xml_create_fpga_cfg_root


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
