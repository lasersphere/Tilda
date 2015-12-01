"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import os
import pickle

import numpy as np

from Service.FileFormat.XmlOperations import xmlCreateIsotope
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


def createXmlFileOneIsotope(scanDict):
    """
    creates an .xml file for one Isotope. Using the Filestructure as stated in OneNote.
    :param scanDict: {'isotopeData', 'activeTrackPar', 'pipeInternals'}
    :return:str, filename
    """
    isodict = scanDict['isotopeData']
    root = xmlCreateIsotope(isodict)
    filename = nameFileXml(scanDict)
    print('creating .xml File: ' + filename)
    save_xml(root, filename, False)
    return filename


def nameFileXml(scanDict):
    """
    finds a filename for the xml file
    :param scanDict: {'isotopeData', 'activeTrackPar', 'pipeInternals'}
    :return:str, filename
    """
    path = scanDict['pipeInternals']['workingDirectory']
    nIso = scanDict['isotopeData']['isotope']
    type = scanDict['isotopeData']['type']
    filename = nameFile(path, 'sums', nIso, str(type + '_sum'), '.xml')
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
    # print('saving: ' + str(data) + ' , to: ' + str(path))
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()
    return path


def loadPickle(file):
    """
    loads the content of a binary file using the pickle module.
    """
    stream = open(file, 'rb')
    data = pickle.load(stream)
    stream.close()
    return data


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