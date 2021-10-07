'''
Created on 3 Nov 2017

@author: andrew
'''
listAtoms = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Fe','Zn','S','I','Br','Cl']
import csv
from yaferp.interfaces import fermilibInterface
from datetime import datetime
from yaferp.analysis.analyserNew import storeInDict
from yaferp.general import PATHS
import os

XYZLOC = '/home/andrew/workspace/BKDataNEW/fermiLibInput/xyz/'
def readListMols(filePath='/home/andrew/workspace/BKDataNEW/fermiLibMols/molecules2.csv'):
    with open(filePath,'r') as f:
        fred = csv.DictReader(f)
        listMols = []
        for row in fred:
            listMols.append(row)
    return listMols

def generateFermiLibMolecule(molDataDict,symmetry='c1'):
    geomLoc = XYZLOC + molDataDict['xyzloc'] + '.xyz'
    basis = molDataDict['basis']
    charge = int(molDataDict['charge'])
    multiplicity = int(molDataDict['multiplicity'])
    description = molDataDict['Name']
    
    fred = fermilibInterface.FermiLibMolecule(geomLoc, basis, multiplicity, charge, description,description,symmetry)
    fred.save()
    return fred

def genLotsOfFLM(filePath='/home/andrew/workspace/BKDataNEW/fermiLibMols/molecules2.csv',runPSI4=1,order=1,symmetry='c1'):
    listMolDataDicts = readListMols(filePath)
    results = []
    for molData in listMolDataDicts:
        print('attempting PSI4 {}'.format(molData))
        try:
            fred = generateFermiLibMolecule(molData,symmetry)
            if runPSI4:
                fred.runPSI4()
            results.append(fred)
        except:
            print('ERROR!')
            print(molData)
            raise
    if order:
        return orderByNumOrbitals(results)
    else:
        return results

def saveLotsOfHamiltonians(filePath='/home/andrew/workspace/BKDataNEW/fermiLibMols/molecules2.csv', overrideCutoff=1e-12,protocol='cpickle',verbose=False,symmetry='c1',boolJWorBK=0):
    listMolObjects = genLotsOfFLM(filePath,symmetry=symmetry)
    for index, molObject in enumerate(listMolObjects):
        print('{} doing {} of {}'.format(str(datetime.now()), str(index),str(len(listMolObjects))))
        molObject.saveHamiltonian(boolJWorBK,overrideCutoff,protocol=protocol,verbose=verbose)
    return

def saveLotsOfIntegrals(filePath='/home/andrew/workspace/BKDataNEW/fermiLibMols/molecules2.csv', overrideCutoff=1e-14,protocol='cpickle',verbose=False):
    listMolObjects = genLotsOfFLM(filePath)
    for index, molObject in enumerate(listMolObjects):
        print('{} doing {} of {}'.format(str(datetime.now()), str(index),str(len(listMolObjects))))
        molObject.saveIntegrals(overrideCutoff)
    return

def orderByNumOrbitals(listFermiMOS):
    return sorted(listFermiMOS, key=lambda x: x.numMOs())

def printNumOrbitals(listFermiMOS):
    for thing in listFermiMOS:
        print(thing.numMOs())
    return




def genAtomXYZ():
    for atom in listAtoms:
        with open('/home/andrew/workspace/BKDataNEW/fermiLibInput/xyz/'+ atom + '.xyz','wb') as f:
            f.write(atom + " 0. 0. 0.")
    return
'''
def fci(fermiLibData):
    flm = generateFermiLibMolecule(fermiLibData)
    flm.runPSI4(runSCF=1,runFCI=1)
    fciEnergy = flm.energyFCI()
    nucRepulsion = flm.energyNuclear()
    return (fciEnergy-nucRepulsion)
'''

def fci(fermiLibMol):
    fermiLibMol.runPSI4(runSCF=1,runFCI=1)
    fciEnergy = fermiLibMol.energyFCI()
    nucRepulsion = fermiLibMol.energyNuclear()
    return(fciEnergy-nucRepulsion)
    
def manyFCI(filePath='/home/andrew/workspace/BKDataNEW/fermiLibMols/molecules2.csv'):
    os.environ['PSI_SCRATCH'] = '/home/andrew/workspace/scratch/PSI4SCR/'
    fred = genLotsOfFLM(filePath)
    reducedFred = [x for x in fred if x.numMOs() <= 40 and x.numMOs() > 30]
    for thing in reducedFred:
        try:
            energyPath = PATHS.ENERGY_DIR + '/1e-12/' + thing.name + '.egs'
            clive = fci(thing)
            storeInDict(energyPath,'FCI',clive)
            storeInDict(energyPath,'NUCLEAR REPULSION', thing.energyNuclear())
        except:
            print('noconv ' + thing.name)
    return

if __name__ == '__main__':
    pass
    #STO-3G bulk indexed by CAS number
    
    #atoms

    #H2 & HeH+ many basis
    
    
    #extras 
