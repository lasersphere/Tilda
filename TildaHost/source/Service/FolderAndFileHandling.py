"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import os

def FindTildaFolder(path=os.path.dirname(os.path.abspath(__file__))):
    if 'Tilda' in os.path.split(path)[0]:
        path = FindTildaFolder(os.path.dirname(path))
    elif 'Tilda' in os.path.split(path)[1]:
        path = path
    else:
        path = 'could not find Tilda folder'
    return path


