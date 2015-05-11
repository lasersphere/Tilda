"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import os



class FolderAndFileHandling(object):
    def __init__(self):
        pass

    def FindTildaFolder(self, path):
        if 'Tilda' in os.path.split(path)[0]:
            path = self.FindTildaFolder(os.path.dirname(path))
        elif 'Tilda' in os.path.split(path)[1]:
            path = path
        else:
            path = 'could not find Tilda folder'
        return path


