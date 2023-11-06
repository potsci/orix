"""Reader of a crystal map from an .osc file in formats produced by EDAX
    This code is a conversion from the mtex load_osc file
"""

import numpy as np

def loadEBSD_osc(fname,**kwargs):
    ebsd = EBSD()

    assert fname.endswith('.osc'), "File must have the .osc extension."

    CS = kwargs.get('CS', oscHeader(fname))

    if kwargs.get('check'):
        return

    data, Xstep, Ystep = oscData(fname)

    nCols = data.shape[1]

    colNames = ['phi1', 'Phi', 'phi2', 'x', 'y', 'ImageQuality', 'ConfidenceIndex', 'Phase', 'SemSignal', 'Fit']

    if nCols > 10:
        for col in range(len(colNames) + 1, nCols):
            colNames.append(f'unknown_{col}')
        print('Warning: More column data was passed in than expected. Check your column names make sense!')
    elif nCols < 5:
        raise ValueError('Error: Need to pass in at least position and orientation data.')
    elif nCols < 9:
        print('Warning: Less column data was passed in than expected. Check your column names make sense.')

    loader = loadHelper(data, ColumnNames=colNames[:nCols], Radians=True)

    if Xstep != Ystep:  # probably hexagonal
        unitCell = np.array([
            [-Xstep/2, -Ystep/3],
            [-Xstep/2, Ystep/3],
            [0, 2*Ystep/3],
            [Xstep/2, Ystep/3],
            [Xstep/2, -Ystep/3],
            [0, -2*Ystep/3]
        ])
    else:
        unitCell = np.array([
            [Xstep/2, -Ystep/2],
            [Xstep/2, Ystep/2],
            [-Xstep/2, Ystep/2],
            [-Xstep/2, -Ystep/2]
        ])

    ebsd = EBSD(loader.getRotations(), loader.getColumnData('phase'), CS,
                loader.getOptions('ignoreColumns', 'phase'), unitCell=unitCell)

    rot = [
        rotation.byAxisAngle(xvector + yvector, 180 * degree),
        rotation.byAxisAngle(xvector - yvector, 180 * degree),
        rotation.byAxisAngle(xvector, 180 * degree),
        rotation.byAxisAngle(yvector, 180 * degree)
    ]

    corSettings = ['notSet', 'setting 1', 'setting 2', 'setting 3', 'setting 4']
    corSetting = get_flag(kwargs, corSettings, 'notSet')
    corSetting = corSettings.index(corSetting.lower()) - 1

    if kwargs.get('convertSpatial2EulerReferenceFrame'):
        flag = 'keepEuler'
        opt = 'convertSpatial2EulerReferenceFrame'
    elif kwargs.get('convertEuler2SpatialReferenceFrame'):
        flag = 'keepXY'
        opt = 'convertEuler2SpatialReferenceFrame'
    else:
        if not kwargs.get('wizard'):
            print('Warning: .ang files usually have inconsistent conventions for spatial coordinates and Euler angles. You may want to use one of the options "convertSpatial2EulerReferenceFrame" or "convertEuler2SpatialReferenceFrame" to correct for this.')
        return

    if corSetting == 0:
        print(f'{opt} was specified, but the reference system setting has not been specified. Assuming "setting 1". Be careful, the default setting of EDAX is "setting 2".')
        print(f'Please make sure you have chosen the correct setting and specify it explicitly using the syntax:\n'
              f'ebsd = EBSD.load(fileName, "{opt}", "setting 2")')
        corSetting = 1

    ebsd = rotate(ebsd, rot[corSetting], flag)

    # The remaining functions oscData and oscHeader are not provided here but can be implemented similarly in Python.

# You will need to implement the oscData and oscHeader functions in Python as well.


def oscHeader(file):
   bufferLength = 2**20
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8, count=bufferLength)

    # Locate the header start and stop markers
    headerStart = np.array([185, 11, 239, 255, 1, 0, 0, 0], dtype=np.uint8)
    headerStop = np.array([185, 11, 239, 255, 2, 0, 0, 0], dtype=np.uint8)
    startBytes_str=headerStart.astype(str)
    startBytes_str=np.char.zfill(startBytes_str, 4)
    startBytes_str=''.join(startBytes_str)

    data_str=data.astype(str)
    data_str=np.char.zfill(data_str, 4)
    data_str=''.join(data_str)
    stopBytes_str=headerStop.astype(str)
    stopBytes_str=np.char.zfill(stopBytes_str, 4)
    stopBytes_str=''.join(stopBytes_str)
    headerStartIndices=int(data_str.find(startBytes_str)/4)
    headerStopIndices=int(data_str.find(stopBytes_str)/4 - 1)

    # headerStartIndices = np.where(np.all(data == headerStart, axis=1))[0]
    # headerStopIndices = np.where(np.all(data == headerStop, axis=1))[0] - 1

    headerBytes = data[headerStartIndices + 8:headerStopIndices]

    # Define the list of known phases and initialize variables
    osc_phases = ['Mg']

    nPhase = 0
    PhaseStart = []
    PhaseName = []
    headerBytes_str="".join([chr(item) for item in headerBytes])
    # Extract phase information 
    # better use regex to find any viable phase names
    for i, phase in enumerate(osc_phases):
        phaseBytes = bytes(phase, 'utf-8')
        phaseLoc = bytes(headerBytes_str,'utf-8').find(phaseBytes)
        
        if phaseLoc != -1:
            nPhase += 1
            PhaseStart.append(phaseLoc + 1)
            PhaseName.append(osc_phases[i])

    if nPhase == 0:
        for i, phase in enumerate(osc_phases):
            phaseBytes = bytes(phase, 'utf-8')
            phaseLoc = headerBytes.find(phaseBytes)
            
            if phaseLoc != -1:
                nPhase += 1
                PhaseStart.append(phaseLoc + 1)
                PhaseName.append(osc_phases[i])

    CS = [None] * nPhase

    # Extract crystal symmetry information
    for k in range(nPhase):
        phaseBytes = headerBytes[PhaseStart[k]:PhaseStart[k] + 288]
        laueGroup = str(np.frombuffer(phaseBytes[256:260], dtype=np.int32)[0])
        cellBytes = phaseBytes[260:284]
        axLength = np.frombuffer(cellBytes[0:12], dtype=np.float32)
        axAngle = np.frombuffer(cellBytes[12:], dtype=np.float32) * np.deg2rad(1)
        numHKL = np.frombuffer(phaseBytes[284:], dtype=np.int32)[0]
        
        options = []
        
        if laueGroup in ['126']:
            laueGroup = '622'
            options.append('X||a')
        elif laueGroup in ['-3m', '32', '3', '62', '6']:
            options.append('X||a')
        elif laueGroup == '2':
            options.append('X||a*')
            print('Warning: symmetry not yet supported!')
        elif laueGroup == '1':
            options.append('X||a')
        elif laueGroup == '131':
            laueGroup = '432'
        elif laueGroup == '20':
            laueGroup = '2'
            options.append('X||a')
        elif np.any(axAngle != np.pi / 2):
            options.append('X||a')
        
        CS[k] = (laueGroup, axLength, axAngle, PhaseName[k], options)

    return CS

    def OSCData(file):
    file='map20231031110414777.osc'
    hexArray=['B9','0B','EF','FF','02','00','00','00']
    startBytes= np.array([int(hexNum, 16) for hexNum in list(hexArray)])
        # Open the file and read the header
    with open(file, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=8)
        # n = header[7]
        n=10
        
        # Set the default start position and buffer length
        startPos = 0
        bufferLength = 2**20
        
        # Read data from the file
        f.seek(startPos, 1)
        startData = np.fromfile(f, dtype=np.uint8, count=bufferLength)
        
        # Find the position of the startBytes pattern
        # startPos = startPos + np.where(np.all(startData == startBytes, axis=1))[0][0]
        startBytes_str=startBytes.astype(str)
        startData_str=startData.astype(str)
        startData_str=''.join(startData_str)
        startBytes_str=''.join(startBytes_str)
        startindex=startData_str.find(startBytes_str)
        startpos=startpos+startindex/2
        # Move to the position after startBytes
        f.seek(startPos + 8, 0)
        
        # Check for different versions of the .osc file
        dn = np.fromfile(f, dtype=np.uint32, count=1)
        if np.round(((dn[0] / 4 - 2) / 10) / n) != 1:
            f.seek(startPos + 8, 0)
        
        # Collect the Xstep and Ystep values
        Xstep = np.fromfile(f, dtype=np.float32, count=1)[0]
        if Xstep == 0:
            Xstep = np.fromfile(f, dtype=np.float32, count=1)[0]
            Ystep = np.fromfile(f, dtype=np.float32, count=1)[0]
        else:
            Ystep = np.fromfile(f, dtype=np.float32, count=1)[0]
        
        # Read the data columns into an array
        position = f.tell()
        for i in range(5, 31):
            data = np.fromfile(f, dtype=np.float32, count=n * i)
            data = data.reshape((n, i))
            data = data.T
            
            # Check if Xstep and Ystep match the data
            if np.isclose(data[1, 4], Xstep, rtol=1e-4) and np.isclose(data[1, 5], 0, rtol=1e-4):
                break
            assert i != 30, "Max number of columns reached, format not handled"
            
            f.seek(position, 0)

    return data, Xstep, Ystep