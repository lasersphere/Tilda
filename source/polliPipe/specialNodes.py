"""
Created on 28.03.2015

@author: dropy
"""
from polliPipe.node import Node


class MergeNode(Node):
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

class CompoundNode(Node):
    """
    This node is a compound of several nodes. Only linear constructs are allowed!
    """
    def __init__(self, first):
        """
        :return:
        """
        super(CompoundNode, self).__init__()
        #self.next is set for compliance with standard node behaviour. self.first should be used.
        self.first = first
        self.next = [first]
        self.last = first
        while self.last.next:
            self.last = self.last.next[0]

    def attach(self, _next):
        """
        Attach _next to the end of the compound node
        :param _next:
        :return:
        """
        self.last.next.attach(next)

    def processItem(self, item):
        """
        Process the item by handing it to the first node.
        This leads to the compound container "vanishing" during execution.
        :param item:
        :return:
        """
        return self.first.processItem(item)
