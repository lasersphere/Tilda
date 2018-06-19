"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import os
import pickle
import json
import h5py
from copy import deepcopy

import numpy as np

import TildaTools as Tits
from Application import Config as Cfg
from TildaTools import nameFile
from XmlOperations import xml_create_autostart_root, xmlWriteDict, xml_create_fpga_cfg_root


def findTildaFolder(path=os.path.dirname(os.path.abspath(__file__))):
    """
    tries to find the Tilda folder relative to execution file
    :return: str, path of Tilda Folder
    """
    if 'TildaHost' == os.path.basename(path):
        return os.path.dirname(path)
    elif os.path.basename(path) == '':
        path = 'could not find Tilda folder'
        return path
    else:
        return findTildaFolder(os.path.split(path)[0])


def delete_file(file):
    """ this will delete an existing file """
    if os.path.isfile(file):
        logging.info('deleting file: %s' % file)
        os.remove(file)


""" pickle related (will not be used anymore in the future) """


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


""" JSON related """


def save_pipedata_json(pipe_data_dict, ending='.jpipedat'):
    """ use this to srialize the pipedata dict and store it in a .jpipedat text file """
    pipe_data_dict = deepcopy(pipe_data_dict)
    path = pipe_data_dict['pipeInternals']['workingDirectory']
    xml_path = os.path.split(pipe_data_dict['pipeInternals']['activeXmlFilePath'])[1].split('.')[0]
    path = nameFile(path, 'raw',
                    xml_path,
                    '',
                    ending)
    for key, val in pipe_data_dict.items():
        if 'track' in key:
            #  Trigger enum needs to be converted to string
            val['trigger']['type'] = val['trigger']['type'].name
    save_json_simple(path, pipe_data_dict)
    return path


def save_json_simple(path, data_to_save):
    """ any data serializable with json will be stored in a textfile to path. """
    try:
        with open(path, 'w') as f:
            json.dump(data_to_save, f)
    except Exception as e:
        logging.error('error json could not dump data, error is: %s' % e, exc_info=True)
    return path


def load_json(path, convert_trigger=True):
    """ load a textfile whihc was serialized using json. """
    with open(path, 'r') as f:
        ret = json.load(f)
    if convert_trigger:
        from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TriTyp
        for key, val in ret.items():
            if 'track' in key:
                val['trigger']['type'] = TriTyp[val['trigger']['type']]
    return ret


""" hdf5/h5py related """


def save_hdf5(path, data, set_name='raw_data'):
    """
    store large sets of data, (mostly numpy arrays probably) Other supported datatypes:
    Type 	Precisions 	Notes
    Integer 	1, 2, 4 or 8 byte, BE/LE, signed/unsigned
    Float 	2, 4, 8, 12, 16 byte, BE/LE
    Complex 	8 or 16 byte, BE/LE 	Stored as HDF5 struct
    Compound 	Arbitrary names and offsets
    Strings (fixed-length) 	Any length
    Strings (variable-length) 	Any length, ASCII or Unicode
    Opaque (kind ‘V’) 	Any length
    Boolean 	NumPy 1-byte bool 	Stored as HDF5 enum
    Array 	Any supported type
    Enumeration 	Any NumPy integer type 	Read/write as integers
    References 	Region and object
    Variable length array 	Any supported type 	See Special Types

    advantage vs. pickle is that hdf5 is universal and can be read by other programming languages, e.g. C...
    also for large datasets (>50000 elements) it seems a bit faster.
    """
    f_raw = h5py.File(path, 'w')
    f_raw.create_dataset(set_name, data=data)
    f_raw.close()
    return path


def load_hdf5(path):
    """ load data from an hdf5 file -> returns the data, e.g. numpy array """
    f_load = h5py.File(path, 'r')
    ret = f_load['raw_data'].value
    f_load.close()
    return ret


def save_raw_data_hdf5(pipe_data_dict, data, ending='.hdf5'):
    """ save the raw data (np.array) from the fpga to a hdf5 file """
    path = pipe_data_dict['pipeInternals']['workingDirectory']
    xml_path = os.path.split(pipe_data_dict['pipeInternals']['activeXmlFilePath'])[1].split('.')[0]
    path = nameFile(path, 'raw',
                    xml_path,
                    '',
                    ending)
    save_hdf5(path, data)
    return path


""" raw data related """


def saveRawData(data, pipeData, nOfSaves):
    """
    function to save the raw data using pickle.
    :return nOfSaves
    """
    if np.count_nonzero(data) > 0:
        # savedto = savePickle(data, pipeData)  # outdated
        savedto = save_raw_data_hdf5(pipeData, data)
        nOfSaves += 1
        logging.info('saving raw data to: ' + str(savedto))
    return nOfSaves


def savePipeData(pipeData, nOfSaves):
    """ save the pipe data dictionary which contains all scan informations """
    savedto = save_pipedata_json(pipeData)
    logging.info('saving pipe data to: ' + str(savedto))
    nOfSaves += 1
    return nOfSaves


""" autostart related: """


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
        root = xml_create_autostart_root(Cfg.version)
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


""" fpga xml config file related """


def load_fpga_xml_config_file():
    """
    load/create the xml config file which is located at

        ...\Tilda\TildaHost\source\Driver\DataAcquisitionFpga\fpga_config.xml

    -> This holds the type(s) of installed fpga's and the "location"/resource (e.g. Rio0)
    """
    # print('loading fpga cfg')
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


""" text file operations """


def save_txt_file_line_by_line(path, list_of_lines_str):
    """
    will save to an textfile as specified in path.
    :param path: str, path, ending should be .txt
    :param list_of_lines_str: list of str, each str will eb written to new line.
    """
    if not os.path.isdir(os.path.basename(path)):
        os.mkdir(os.path.basename(path))
    if '.txt' not in path:
        path += '.txt'
    with open(path, 'w') as txt_file:
        for each in list_of_lines_str:
            txt_file.write(each + '\n')
    txt_file.close()


def load_from_text_file(path):
    """
    will return a list with each item as a line in the textfile specified at path.
    :param path: str, path, ending should be .txt
    :return: list, of strings, each one is a line in the textfile at path
    """
    if os.path.isfile(path):
        with open(path, 'r') as txt_file:
            data = []
            line = txt_file.readline()[:-1]
            while line:
                data.append(line)
                line = txt_file.readline()[:-1]
        txt_file.close()
        return data
    else:
        return []
