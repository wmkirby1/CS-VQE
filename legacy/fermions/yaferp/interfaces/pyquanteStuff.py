'''
Created on 11 Nov 2014

@author: andrew
'''

from PyQuante import Molecule
from PyQuante.Basis.sto3g import basis_data
from PyQuante.Ints import getints,getbasis
methane = Molecule('CH4',
                    [(6, (0.,    0.,    0.)),
                    (1, (0.639648,  0.639648,    0.639648)),
                    (1, (-0.639648,    -0.639648,    0.639648)),
                    (1, (-0.639648,    0.639648,    -0.639648)),
                    (1, (0.639648,    -0.639648,    -0.639648))],
                   units='Angstrom')
en,orbe,orbs = rhf(methane,basis_data=basis_data)
basis = getbasis(methane,basis_data)
overlap,oei,tei = getints(basis,methane)