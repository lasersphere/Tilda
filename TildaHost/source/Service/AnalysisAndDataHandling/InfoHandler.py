"""
Created on 

@author: simkaufm

Module Description: Module to hanlde the information coming from the fpga,

32-bit input must begin with 'firstHeader' = 0100 and 'headerIndex' = 1
"""


class InfoHandler:
    def __init__(self):
        self.completed_bunches_in_step = None

    def setup(self):
        if self.completed_bunches_in_step is None:
            self.completed_bunches_in_step = 0

    def clear(self):
        self.completed_bunches_in_step = None

    def info_handle(self, pipe_data, payload):
        """
        will directly write into pipeData
        :param pipe_data:
        :param payload:
        """
        track_ind, track_name = pipe_data['pipeInternals']['activeTrackNumber']
        if payload == 1:  # means step complete
            bun = pipe_data[track_name].get('nOfBunches', -1)
            # step will always be complete if 'nOfBunches' cannot be found, e.g. in cont seq
            self.completed_bunches_in_step += 1
            if self.completed_bunches_in_step == bun or bun < 0:
                self.completed_bunches_in_step = 0
                comp_steps = pipe_data[track_name]['nOfCompletedSteps'] + 1
                steps = pipe_data[track_name]['nOfSteps']
                volt_index = comp_steps % steps
                pipe_data[track_name]['nOfCompletedSteps'] = comp_steps
                return volt_index

        elif payload == 2:  # means scan complete
            pipe_data[track_name]['nOfScans'] += 1
            volt_index = 0
            return volt_index

        elif payload == 3:  # means new bunch
            return None
