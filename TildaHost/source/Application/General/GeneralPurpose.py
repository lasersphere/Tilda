"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import os



class GeneralPurpose(object):
    def __init__(self):
        pass

    def ResolveBitfileLocation(self, folder, dllName):
        path = os.path.join(self.FindTildaFolder(os.getcwd()), 'TildaTarget\\bin\\' ,folder, dllName)
        if not os.path.isfile(path):
            path = 'file not found'
        return path


    def FindTildaFolder(self, path):
        if 'Tilda' in os.path.split(path)[0]:
            path = self.FindTildaFolder(os.path.dirname(path))
        elif 'Tilda' in os.path.split(path)[1]:
            path = path
        else:
            path = 'could not find Tilda folder'
        return path


