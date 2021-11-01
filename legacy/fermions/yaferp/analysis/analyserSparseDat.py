from yaferp.analysis.analyserHPCNew import loadOplist,getMoleculesInDirectory
import pickle
import yaferp.general.sparseLinalg
import datetime
DATA_DIR = '/home/andrew/data/BKData/'
SPARSE_DAT_DIR = DATA_DIR + 'hamiltonian/sparseDat/'
MAPPING_BOOL_TO_STRING = ['JW','BK']
def sparseGenerator(filename, boolJWorBK, cutoff=1e-12):
    oplist = loadOplist(filename,boolJWorBK,cutoff,ordering='magnitude')
    outputPath = '{}{}/magnitude/{}/{}.sdat'.format(SPARSE_DAT_DIR,str(cutoff),str(MAPPING_BOOL_TO_STRING[boolJWorBK]),filename)
    result = yaferp.general.sparseLinalg.oplistToSparseData(oplist)
    with open(outputPath,'wb') as f:
        pickle.dump(result,f)
    return

def sparseGeneratorAll(directory,boolJWorBK,cutoff=1e-12):
    filenames = getMoleculesInDirectory(directory)
    lastTime = datetime.datetime.now()
    for i,filename in enumerate(filenames):
        sparseGenerator(filename,boolJWorBK,cutoff)
        currentTime = datetime.datetime.now()
        print('{}   Done {}, {}/{} complete.  Time taken:  {}'.format(currentTime,filename,i+1,len(filenames),currentTime-lastTime))
        lastTime=currentTime
    return