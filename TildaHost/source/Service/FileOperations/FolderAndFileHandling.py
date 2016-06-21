"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import os
import pickle
from copy import deepcopy

import numpy as np

from Service.FileOperations.XmlOperations import xmlCreateIsotope, xml_add_meas_volt_pars
from TildaTools import save_xml
import Tools


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
        Tools._insertFile(filename, db)
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
    path = nameFile(path, 'raw',
                    pipeDataDict['isotopeData']['isotope'] +
                    '_track' + str(pipeDataDict['pipeInternals']['activeTrackNumber'][0]),
                    pipeDataDict['isotopeData']['type'],
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