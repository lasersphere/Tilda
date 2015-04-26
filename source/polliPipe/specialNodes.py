"""
Created on 28.03.2015

@author: dropy
"""
from polliPipe import node


class MergeNode(node):
    """
    This node is capable of merging multiple items into one
    """


    def __init__(self, nInp):
        """
        Create a node to merge nInp incoming item streams
        """
        super(MergeNode, self).__init__()
        
        
    def processData(self, a):
        """
        This accumulates the different items and creates a merged one if all have been handed over
        """
        pass
        