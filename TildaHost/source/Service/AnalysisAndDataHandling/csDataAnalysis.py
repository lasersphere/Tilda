"""

Created on '09.07.2015'

@author:'simkaufm'

"""

def checkIfScanComplete(pipeData):
    """
    Check if all Steps for this scan have been completed
    :return: bool, True if one Scan over all steps is completed, transfer Data to next node.
    """
    return pipeData['activeTrackPar']['nOfCompletedSteps'] % pipeData['activeTrackPar']['nOfSteps'] == 0

def checkIfTrackComplete(pipeData):
    """
    Check if all Steps for this track have been completed
    :return: bool, True if finished
    """
    nOfCompletedSteps = pipeData['activeTrackPar']['nOfCompletedSteps']
    nOfScans = pipeData['activeTrackPar']['nOfScans']
    nOfSteps = pipeData['activeTrackPar']['nOfSteps']
    return nOfCompletedSteps == nOfScans * nOfSteps