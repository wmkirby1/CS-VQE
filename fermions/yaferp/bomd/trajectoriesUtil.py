def getGaussianTrajectories(filePath):
    xyzLines = []

    with open(filePath, 'r') as f:
        line = f.readline()
        while line:
            if "Summary information for step" in line:
                thisRaw = []
                while "Cartesian coordinates" not in line:
                    line = f.readline()
                line = f.readline()
                while "MW cartesian velocity" not in line:
                    thisRaw.append(line)
                    line = f.readline()

                thisProc = [x.strip().split() for x in thisRaw]
                thisProc2 = [[float(x[i].replace('D','E')) for i in [3,5,7]]for x in thisProc]
                newLines = [str(len(thisProc2)),'']
                newLines += ['H {} {} {}'.format(x[0],x[1],x[2]) for x in thisProc2]
                xyzLines += newLines

            line = f.readline()

                
    return (xyzLines)

def convertGaussianToXYZ(filePath):
    output = getGaussianTrajectories(filePath)
    outputPath = '.'.join(filePath.split('.')[:-1]) + '.xyz'
    with open(outputPath,'w') as f:
        f.writelines([x + '\n' for x in output])
    return
