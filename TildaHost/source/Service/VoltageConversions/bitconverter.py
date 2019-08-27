
class BitConverter:


    def get_max_value_in_bits(self,):
        return None

    def get_bits_from_voltage(self,voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
       return None


    def get_stepsize_in_bits(self,step_voltage, dac_gauge_pars):
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