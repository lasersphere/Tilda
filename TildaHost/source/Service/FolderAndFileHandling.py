"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import lxml.etree as ET
import os
import pickle
import time
import Service.Formating as form

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
    find an unused valid filename. Add Timestamp in front.
    :return: str, path/subdir/timestamp_prefix_fileName + suffix
    """
    storagePath = os.path.join(path, subdir)
    if not os.path.exists(storagePath):
        os.makedirs(storagePath)
    filepath = os.path.join(storagePath, time.strftime("%Y%m%d_%H%M%S") + '_' +
                            prefix + '_' + fileName)
    i = 0
    if not os.path.isfile(filepath + suffix):
        return filepath + suffix
    while os.path.isfile(filepath +  '_' + str(i) + suffix):
        i += 1
    return filepath + '_' + str(i) + suffix

def createXmlFileOneIsotope(scanDict):
    isodict = scanDict['isotopeData']
    path = isodict['pathFile']
    form.xmlCreateIsotope()

def saveXml(rootEle, filename, pretty=True):
    """
    Convert a Root lxml Element into an ElementTree and save it to a file
    """
    tree = ET.ElementTree(rootEle)
    tree.write(filename, pretty_print=pretty)

def loadXml(filename):
    """
    loads an .xml file and returns it as an lxml.etree.Element
    :return:lxml.etree.Element, Element of loaded File
    """
    tree = ET.parse(filename)
    elem = tree.getroot()
    return elem

def savePickle(data, pipeDataDict):
    """
    saves data using the pickle module
    """
    path = pipeDataDict['pipeInternals']['filePath']
    path = nameFile(path, 'raw',
                    pipeDataDict['isotopeData']['isotope'] +
                    '_track' + str(pipeDataDict['pipeInternals']['activeTrackNumber']),
                    pipeDataDict['isotopeData']['type'],
                    '.raw')
    # print('saving: ' + str(data) + ' , to: ' + str(path))
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def loadPickle(file):
    """
    loads the content of a binary file using the pickle module.
    """
    stream = open(file, 'rb')
    data = pickle.load(stream)
    stream.close()
    return data