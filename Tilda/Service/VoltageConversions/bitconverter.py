
class BitConverter:


    def get_max_value_in_bits(self,):
        return None

    def get_nbits_from_voltage(self, voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
       return None

    def get_20bits_from_voltage(self, voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
        return None

    def get_nbit_stepsize(self, step_voltage, dac_gauge_pars):
      return None

    def get_20bit_stepsize(self, step_voltage, dac_gauge_pars):
        return None

    def get_stepsize_in_volt_from_bits(self,voltage_bits, dac_gauge_pars):
       return None


    def get_24bit_input_from_voltage(self,voltage, dac_gauge_pars,
                                     add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
      return None


    def get_voltage_from_bits(self,voltage_20bit, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
       return None


    def get_voltage_from_24bit(self,voltage_24bit, dac_gauge_pars,
                               remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
       return None


    def get_bits_from_24bit_dac_reg(self,voltage_24bit, remove_address=True):
       return None

    def calc_step_size(self, start, stop, steps):
        """
        calculates the stepsize: (stop - start) / nOfSteps
        :return stepsize_18bit
        """
        try:
            dis = stop - start
            stepsize_18bit = int(dis / (steps - 1))
        except ZeroDivisionError:
            stepsize_18bit = 0
        # stepsize_18bit = max(-(2 ** 18 - 1), stepsize_18bit)
        # stepsize_18bit = min((2 ** 18 - 1), stepsize_18bit)
        return stepsize_18bit

    def calc_n_of_steps(self, start, stop, step_size):
        """
        calculates the number of steps: abs((stop - start) / stepSize)
        """
        try:
            dis = abs(stop - start) + abs(step_size)
            n_of_steps = int(dis / abs(step_size))
        except ZeroDivisionError:
            n_of_steps = 0
        # n_of_steps = max(2, n_of_steps)
        # n_of_steps = min((2 ** 18 - 1), n_of_steps)
        return n_of_steps