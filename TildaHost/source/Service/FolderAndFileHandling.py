"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import lxml.etree as ET
import os
import pickle

def FindTildaFolder(path=os.path.dirname(os.path.abspath(__file__))):
    """
    tries to find the Tilda folder relative to execution file
    :return: str, path of Tilda Folder
    """
    if 'Tilda' in os.path.split(path)[0]:
        path = FindTildaFolder(os.path.dirname(path))
    elif 'Tilda' in os.path.split(path)[1]:
        path = path
    else:
        path = 'could not find Tilda folder'
    return path

def saveXml(rootEle, filename, pretty=True):
    """
    Convert a Root lxml Element into an ElementTree and save it to file
    """
    tree = ET.ElementTree(rootEle)
    tree.write(filename, pretty_print = pretty)

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
    :param data:
    :param pipeDataDict:
    :return:
    """
    path = pipeDataDict['pipeInternals']['filePath']
    path = os.path.join(path, 'TestData')
    print('saving: ' + str(data) + ' , to: ' + str(path))
    # pickle.dump(data, pipeDataDict['pipeInternals']['filePath'])