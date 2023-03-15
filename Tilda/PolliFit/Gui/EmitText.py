"""
Created on 10.06.2014

@author: hammen
"""

import sys

from PyQt5 import QtCore


class EmitText(QtCore.QObject):
    """
    classdocs
    """

    textSig = QtCore.pyqtSignal(str)
        
    def write(self, text):
        sys.__stdout__.write(text)
        self.textSig.emit(text.strip())
