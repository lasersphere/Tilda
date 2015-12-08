"""
Created on 

@author: simkaufm

Module Description: Module for simulating a simple Counter, if no FPGA is at hand
"""

import Service.Formating as Form


import numpy as np

class SimpleCounterDummy:
    def __init__(self):
        pass

    def conf_host_buf(self, num_of_request_ele):
        return True

    def read_data_from_fifo(self):
        """
        returns dummy data which is freshly produced at each call.
        will give 100 elements each call
        :return: {nOfEle,  newData, elemRemainInFifo}
        nOfEle = int, number of Read Elements,
        newData = numpy.ndarray containing all data that was read
        elemRemainInFifo = int, number of Elements still in FifoBuffer
        """
        read_dict = dict.fromkeys(['nOfEle', 'newData', 'elemRemainInFifo'])
        n_ele = 10
        read_dict['nOfEle'] = n_ele * 8
        read_dict['newData'] = self.dummy_data(n_ele)
        read_dict['elemRemainInFifo'] = 0
        return read_dict

    def set_dac_voltage(self, volt_dbl):
        return True

    def set_post_acc_control(self, state_name):
        pass

    def get_post_acc_control(self):
        post_acc_state = 0
        post_acc_name = 'Kepco'
        return post_acc_state, post_acc_name

    def dummy_data(self, num_of_vals):
        """
        builds dummy data with form:
        pmt_num + 1
        and sorts it into array like it would come from fpga so:
        (
        (32-bit pmt0), (32-bit pmt1), ..., (32-bit pmt7),
         (32-bit pmt0), (32-bit pmt1), ..., (32-bit pmt7),
          ...
          )
        :parameter: num_of_vals = number of samples for each pmt
        :return: np.array, in 32-Bit data format len(np.array) = num_of_vals * 8
        """
        data = np.zeros((8, num_of_vals), dtype=np.int32)
        for pmt_num in range(0, 8):
            val = Form.add_header_to23_bit(pmt_num + 1, 1, pmt_num, 1)
            data[pmt_num] = np.full((num_of_vals,), val)
        return data.flatten('F')


# scd = SimpleCounterDummy()
# d = scd.dummy_data(100)
# print(d.flatten('F'))