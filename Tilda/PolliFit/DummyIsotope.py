"""
Created on 24.07.2018

@author: simkaufm

Module Description: Dummy isotope to create an isotope from script without the need to have a database
"""

from Tilda.PolliFit import Physics


class DummyIsotope(object):
    """
    Dummy isotope to create an isotope from script without the need to have a database
    """

    def __init__(self, even, iso, isovar='', lineVar='', line_list=None, iso_list=None):
        """
        create an isotope with standard pars or from a list, either even or odd ->  see line_list / iso_list
        :param even: bool, True for an even isotope (will be overwritten if iso_list is not None)
        :param iso: str, name of iso
        :param isovar: str, variable for isotope (adds to the name for flexibility, kept for completion to DummyIsotope)
        :param lineVar: str, variable for a line (kept for completion to DummyIsotope)
        :param line_list: list or None for default line
                else list = [reference_str, frequency_dbl, Jl_dbl, Ju_dbl,
                             shape_str_dict, fixShape_str_dict, charge_int(1/0)]
        :param iso_list: list or None for default iso, see config below
                else list = [mass_dbl, mass_d_dbl, I_dbl, center_dbl,
                             Al_dbl, Bl_dbl, Au_dbl, Bu_dbl,
                             fixedArat_bool, fixedBrat_bool, intScale_dbl, fixedInt_int(0/1),
                             relInt_str_list, m_str, fixedAl_int(0/1), fixedBl_int(0/1),
                             fixedAu_int(0/1), fixedBu_int(0/1)]
        """
        print('iso: ' + str(iso))
        print('isovar: ' + str(isovar))
        print("Loading", lineVar, "line of", iso + isovar)

        if line_list is None:
            # use default settings -> voigt (from Nickel2017)
            line_list = [iso + isovar, 850344226.10401, 3, 2,
                         str({'name': 'Voigt', 'gau': 28, 'lor': 10.57}),
                         # str for better compatibility with DummyIsotope
                         str({'gau': False, 'lor': True}),
                         0
                         ]

        self.name = iso + isovar
        self.isovar = isovar
        self.lineVar = lineVar
        self.ref = line_list[0]
        self.freq = line_list[1]
        self.Jl = line_list[2]
        self.Ju = line_list[3]
        self.shape = eval(line_list[4])
        self.fixShape = eval(line_list[5])
        elmass = line_list[6] * Physics.me_u
        print('loaded :', self.name)

        if iso_list is None:
            if even:
                # use default settings -> 60Ni
                # mass_dbl, mass_d_dbl, I_dbl, center_dbl,
                # Al_dbl, Bl_dbl, Au_dbl, Bu_dbl,
                # fixedArat_bool, fixedBrat_bool, intScale_dbl, fixedInt_int(0/1),
                # relInt_str_list, m_str, fixedAl_int(0/1), fixedBl_int(0/1),
                # fixedAu_int(0/1), fixedBu_int(0/1)
                iso_list = [59.9307859, 5e-07, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0, 0, 4000.0, 0,
                            None, None, 0, 0,
                            0, 0]
            else:
                # use default settings -> 61Ni
                iso_list = [60.9310556, 5e-07, 1.5, 270.0, -454.972, -102.951, -176.8, -47.4,
                            0, 0, 700.0, 0,
                            '[0.4, 0.16, 0.01, 0.64, 0.21, 0.01, 0.98, 0.16, 1.43]', None, 0, 0, 0, 0]

        self.mass = iso_list[0] - elmass
        self.mass_d = iso_list[1]
        self.I = iso_list[2]
        self.center = iso_list[3]
        self.Al = iso_list[4]
        self.Bl = iso_list[5]
        self.Au = iso_list[6]
        self.Bu = iso_list[7]
        self.fixArat = iso_list[8]
        self.fixBrat = iso_list[9]
        self.intScale = iso_list[10]
        self.fixInt = iso_list[11]
        self.fixedAl = bool(iso_list[14])
        self.fixedBl = bool(iso_list[15])
        self.fixedAu = bool(iso_list[16])
        self.fixedBu = bool(iso_list[17])

        if iso_list[12]:
            self.relInt = eval(iso_list[12])
        else:
            self.relInt = []
        if not iso_list[13]:
            self.m = None
        else:
            self.m = DummyIsotope(even, iso_list[13], lineVar)  # load an isomer with default settings if wanted


if __name__ == '__main__':
    iso = DummyIsotope(True, 'tester')
