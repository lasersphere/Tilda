"""
Created on 31.01.2019

@author: pamueller
"""

import Physics

from scipy.integrate import romb
import numpy as np


class ThermalGradientVoigt(object):
    """
    Implementation of a numerically calculated thermal Voigt profile object
    resulting from probing an accelerated ensemble of ions.
    It is assumed that the ions are initially thermal distributed and created along a linear voltage gradient.
    Spatially, the ions follow a normal distribution were the center location corresponds to the zero-point energy.
    The extra spatial distribution is equivalent to normally distributed energies.
    Hence, this profile accounts for both a voltage gradient inside the source region
    as well as scattering of the source potential with a std. deviation of 'sigma' (MHz).

    The distribution of the kinetic energy of the ions after acceleration is calculated analytically.
    It can be written in terms of the modified bessel functions of the first and second kind

    I_(1/4), I_(-1/4) and K_(1/4) (see method Physics.source_energy_pdf).

    The energies are substituted with frequencies. The kinetic energies of the ions approximately depend linearly
    on the frequencies of a fixed laser in the rest frame of the ions.
    The proportionality constant is the parameter 'xi'/2 (MHz / kT)
    For the calculation of 'xi', see [Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)].

    The final lineshape is calculated numerically and is given by
    the expectation value of the Lorentz function given the described frequency distribution.

    The numerical integration is done using Romberg integration.
    A 'precision' parameter is used to control the number (2 ** n_precision + 1) of samples used for integration.
    A precision of 8 seems to be sufficient and yields a relatively fast computation.
    The integration interval is dynamically determined based on the two width parameters 'sigma' and 'xi'.

    The peak height is normalized to one at the 'center' position.
    """
    
    def __init__(self, iso):
        """ Initialize """
        self.iso = iso
        self.nPar = 5
        
        self.pGam = 0
        self.pSigma = 1
        self.pXi = 2
        
        self.pColDirTrue = 3
        self.pPrecision = 4
        
        self.recalc([iso.shape['gamma'], iso.shape['sigma'], iso.shape['xi'],
                     iso.shape['col'], iso.shape.get('precision', 8)])
    
    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        n = 2 ** int(abs(p[self.pPrecision])) + 1
        delta_left = 10 * (p[self.pSigma] + np.abs(p[self.pXi])) if p[self.pColDirTrue] else 10 * p[self.pSigma]
        delta_right = 10 * p[self.pSigma] if p[self.pColDirTrue] else 10 * (p[self.pSigma] + np.abs(p[self.pXi]))
        np_version = np.version.version.split('.')
        if int(np_version[0]) >= 1 and int(np_version[1]) >= 16:
            x_int = np.linspace(x - delta_left, x + delta_right, n).T
        else:
            x_int = np.array([np.linspace(x_i - delta_left, x_i + delta_right, n) for x_i in x])
        y = Physics.source_energy_pdf(x_int, np.expand_dims(x, axis=1), p[self.pSigma], p[self.pXi],
                                      collinear=p[self.pColDirTrue]) * Physics.lorentz(x_int, 0., p[self.pGam])
        int_norm = np.max(y)
        y /= int_norm
        y = romb(y, (delta_left + delta_right) / (n - 1)) * int_norm
        return y / self.norm
    
    def recalc(self, p):
        """ Recalculate the norm factor """
        self.norm = float(self.evaluate(np.zeros(1), p))
    
    def leftEdge(self, p):
        """ Return the left edge of the spectrum in MHz """
        xi_left = 0. if p[self.pColDirTrue] else np.abs(p[self.pXi])
        return -5. * (p[self.pGam] + 2. * np.sqrt(2. * np.log(2.)) * p[self.pSigma] + xi_left)
    
    def rightEdge(self, p):
        """ Return the right edge of the spectrum in MHz """
        xi_right = np.abs(p[self.pXi]) if p[self.pColDirTrue] else 0.
        return 5. * (p[self.pGam] + 2. * np.sqrt(2. * np.log(2.)) * p[self.pSigma] + xi_right)
    
    def getPars(self, pos=0):
        """ Return list of initial parameters and initialize positions """
        self.pGam = pos
        self.pSigma = pos + 1
        self.pXi = pos + 2

        self.pColDirTrue = pos + 3
        self.pPrecision = pos + 4

        return [self.iso.shape['gamma'], self.iso.shape['sigma'], self.iso.shape['xi'],
                self.iso.shape['col'], self.iso.shape.get('precision', 8)]
    
    def getParNames(self):
        """ Return list of the parameter names """
        return ['gamma', 'sigma', 'xi', 'col', 'precision']
    
    def getFixed(self):
        """ Return list of parameters with their fixed-status """
        return [self.iso.fixShape['gamma'], self.iso.fixShape['sigma'], self.iso.fixShape['xi'], True, True]
