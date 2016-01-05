"""

Created on '09.07.2015'

@author:'simkaufm'

"""


def checkIfScanComplete(pipeData, totalnOfScalerEvents, track_name):
    """
    Check if all Steps for this scan have been completed
    :return: bool, True if one Scan over all steps is completed, transfer Data to next node.
    """
    stepComplete = totalnOfScalerEvents % 8 == 0
    complete = pipeData[track_name]['nOfCompletedSteps'] % pipeData[track_name]['nOfSteps'] == 0
    notzero = pipeData[track_name]['nOfCompletedSteps'] != 0
    return stepComplete and complete and notzero


def checkIfTrackComplete(pipeData, track_name):
    """
    Check if all Steps for this track have been completed
    :return: bool, True if finished
    """
    nOfCompletedSteps = pipeData[track_name]['nOfCompletedSteps']
    nOfScans = pipeData[track_name]['nOfScans']
    nOfSteps = pipeData[track_name]['nOfSteps']
    return nOfCompletedSteps == nOfScans * nOfSteps