"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import lxml.etree as ET
import os
import pickle
import time
import Service.Formating as form
import numpy as np
import logging

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
    root = form.xmlCreateIsotope(isodict)
    filename = nameFileXml(scanDict)
    print('creating .xml File: ' + filename)
    saveXml(root, filename, False)
    return filename

def nameFileXml(scanDict):
    """
    finds a filename for the xml file
    :param scanDict: {'isotopeData', 'activeTrackPar', 'pipeInternals'}
    :return:str, filename
    """
    path = scanDict['pipeInternals']['filePath']
    nIso = scanDict['isotopeData']['isotope']
    type = scanDict['isotopeData']['type']
    filename = nameFile(path, 'sums', nIso, str(type + '_sum'), '.xml')
    return filename

def saveXml(rootEle, path, pretty=True):
    """
    Convert a Root lxml Element into an ElementTree and save it to a file
    """
    np.set_printoptions(threshold=np.nan)
    tree = ET.ElementTree(rootEle)
    tree.write(path, pretty_print=pretty)

def loadXml(filename):
    """
    loads an .xml file and returns it as an lxml.etree.Element
    :return:lxml.etree.Element, Element of loaded File
    """
    tree = ET.parse(filename)
    elem = tree.getroot()
    return elem

def scanDictionaryFromXmlFile(xmlFileName, nOfTrack, oldDict):
    """
    creates a Scandictionary with the fom as stated in draftScanParameters
    values are gained from the loaded xmlFile
    :return: dict, Scandictionary gained from the xml file.
    """
    xmlEtree = loadXml(xmlFileName)
    trackdict = form.xmlGetDictFromEle(xmlEtree)[1]['tracks']['track' + str(nOfTrack)]['header']
    isotopedict = form.xmlGetDictFromEle(xmlEtree)[1]['header']
    oldDict['isotopeData'] = isotopedict
    oldDict['activeTrackPar'] = trackdict
    oldDict['pipeInternals'] = {}
    oldDict['pipeInternals']['filePath'] = os.path.split(os.path.split(xmlFileName)[0])[0]
    oldDict['pipeInternals']['curVoltInd'] = 0
    oldDict['pipeInternals']['activeTrackNumber'] = nOfTrack
    oldDict['pipeInternals']['activeXmlFilePath'] = xmlFileName
    for key, val in oldDict.items():
        oldDict[str(key)] = form.convertStrValuesInDict(oldDict[str(key)])
    return oldDict

def savePickle(data, pipeDataDict, ending='.raw'):
    """
    saves data using the pickle module
    """
    path = pipeDataDict['pipeInternals']['filePath']
    path = nameFile(path, 'raw',
                    pipeDataDict['isotopeData']['isotope'] +
                    '_track' + str(pipeDataDict['pipeInternals']['activeTrackNumber']),
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