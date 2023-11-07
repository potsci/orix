
"""Reader of a crystal map from an .ang file in formats produced by EDAX
TSL, NanoMegas ASTAR Index or EMsoft's EMdpmerge program.
"""

from io import TextIOWrapper
import re
from typing import List, Optional, Tuple, Union
import warnings

from diffpy.structure import Lattice, Structure
import numpy as np

from orix import __version__
from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from orix.quaternion.symmetry import point_group_aliases

__all__ = ["file_reader", "file_writer"]

# Plugin description
format_name = "osc"
file_extensions = ["osc"]
writes = True
writes_this = CrystalMap

def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from a file in EDAX TLS's .ang format. The
    map in the input is assumed to be 2D.

    Many vendors produce an .ang file. Supported vendors are:

    * EDAX TSL
    * NanoMegas ASTAR Index
    * EMsoft (from program `EMdpmerge`)
    * orix

    All points satisfying the following criteria are classified as not
    indexed:

    * EDAX TSL: confidence index == -1

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    xmap
        Crystal map.
    """
    # Get file heade
    phase_ids, phase_names, symmetries, lattice_constants = _get_osc_header(filename)
    symmetries=_test_alternative_pointgroups(symmetries)
    structures = []
    for name, abcABG in zip(phase_names, lattice_constants):
        abcABG=abcABG[0].tolist()
        structures.append(Structure(title=name, lattice=Lattice(*abcABG)))

    # Read all file data
    file_data,xstep,ystep = _get_osc_data(filename)
    file_data=file_data.T
    # Get vendor and column names
    n_rows, n_cols = file_data.shape
    print(n_rows,n_cols)
    vendor, column_names = _get_vendor_columns(n_cols)

    # Data needed to create a CrystalMap object
    data_dict = {
        "euler1": None,
        "euler2": None,
        "euler3": None,
        "x": None,
        "y": None,
        "phase_id": None,
        "prop": {},
    }
    for column, name in enumerate(column_names):
        if name in data_dict.keys():
            data_dict[name] = file_data[:, column]
        else:
            data_dict["prop"][name] = file_data[:, column]

    # Add phase list to dictionary
    data_dict["phase_list"] = PhaseList(
        names=phase_names,
        point_groups=symmetries,
        structures=structures,
        ids=phase_ids,
    )

    # Set which data points are not indexed
    # TODO: Add not-indexed convention for INDEX ASTAR
    if vendor in ["orix", "tsl"]:
        not_indexed = data_dict["prop"]["ci"] == -1
        data_dict["phase_id"][not_indexed] = -1

    # Set scan unit
    data_dict["scan_unit"] = "um"

    # Create rotations
    data_dict["rotations"] = Rotation.from_euler(
        np.column_stack(
            (data_dict.pop("euler1"), data_dict.pop("euler2"), data_dict.pop("euler3"))
        ),
    )

    return CrystalMap(**data_dict)


def _get_osc_header(file:str) -> Tuple[List[int], List[str], List[str], List[List[float]]]:
    """ return the contens of the head from an .osc file
    function based on mtex functionallity
    ToDO: implement a way, that we know the phases in advance and do not try to read the header at all
    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    [phaseid]
    [phhasename]
    [pointgroup] (Lauge group).
    [lattice constants]

    """

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

    headerBytes = data[headerStartIndices +8 :headerStopIndices]

    # Define the list of known phases and initialize variables
    #osc_phases = ['Magnesium']
    import re
    import re
    pat = re.compile(r'''
    (?<!\\.{3})[A-z]{1,2}[0-9]{0,4}[ ](?:[A-z]+\d+[A-z]*)*|
    (?<!\\.{3})[A-z]{1}[a-z\s]{3,50}|#beginning big all small or all small
    (?<!\\.{3})[A-Z]{4,50}| # all big
    (?<!\\.{3})(?:[A-z]{1,2})+(?:[A-z]{1,2}[0-9]{1,3}\.{0,1}[0-9]{0,5})+|
    (?<!\\.{3})(?:[A-z]){1,2}(?:[A-z]{1,2}[0-9]{1,2})+
    (?<!\\.{3})[A-z]{1,2}[0-9]{0,4}[ ](?:[A-z]+\d+[A-z]*)*
    #(?<!\\.{2})[[A-z](?:[A-z]{1,3}[0,9]{0,10}){2,10}
    #(?<!\\.{2})[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+''',flags=re.ASCII|re.VERBOSE)        #(?<!\\.{2})[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+''',flags=re.ASCII|re.VERBOSE)
    headerBytes_str="".join([chr(item) for item in headerBytes])
    osc_phases=re.findall(pat,headerBytes_str)
    osc_phases=_is_chemical(osc_phases)
    nPhase = 0
    PhaseStart = []
    PhaseName = []

    # Extract phase information 
    # better use regex to find any viable phase names
    for i, phase in enumerate(osc_phases):

        # it occured that if we match against the strings the results are more accurate
        phase_ord=[str(ord(item)) for item in phase]
        phaseBytes_str=np.char.zfill(phase_ord, 4)
        phaseBytes_str=''.join(phaseBytes_str)
        phaseIndices=int(data_str.find(phaseBytes_str)/4)
        phaseLoc=phaseIndices-headerStartIndices-8
        print(phaseLoc)
        if phaseLoc != -1:
            nPhase += 1
            PhaseStart.append(phaseLoc)
            PhaseName.append(osc_phases[i])

    if nPhase == 0:
        for i, phase in enumerate(osc_phases):
                phaseBytes = bytes(phase, 'utf-8')
                phaseLoc = headerBytes.find(phaseBytes)
                
                if phaseLoc != -1:
                    nPhase += 1
                    PhaseStart.append(phaseLoc)
                    PhaseName.append(osc_phases[i])

    CS = {
        "name": [],
        "point_group": [],
        "lattice_constants": [],
        "id": [],
    }

    # Extract crystal symmetry information
    for k in range(nPhase):
        CS['id'].append(k)
        CS['name'].append(PhaseName[k])
        phaseBytes = headerBytes[PhaseStart[k]+3:PhaseStart[k]+ 288] # not excatly sure where the offset of 3 is coming from
        CS['point_group'].append(str(np.frombuffer(phaseBytes[253:257], dtype=np.int32)[0]))
        cellBytes = phaseBytes[257:281]
        CS['lattice_constants'].append([np.concatenate((np.frombuffer(cellBytes[0:12], dtype=np.float32),np.frombuffer(cellBytes[12:], dtype=np.float32)))])
        print(CS)
        # = np.frombuffer(cellBytes[12:], dtype=np.float32) * np.deg2rad(1)
        # numHKL = np.frombuffer(phaseBytes[284:], dtype=np.int32)[0]
        
        # options = []
        
        # if laueGroup in ['126']:
        #     laueGroup = '622'
        #     options.append('X||a')
        # elif laueGroup in ['-3m', '32', '3', '62', '6']:
        #     options.append('X||a')
        # elif laueGroup == '2':
        #     options.append('X||a*')
        #     print('Warning: symmetry not yet supported!')
        # elif laueGroup == '1':
        #     options.append('X||a')
        # elif laueGroup == '131':
        #     laueGroup = '432'
        # elif laueGroup == '20':
        #     laueGroup = '2'
        #     options.append('X||a')
        # elif np.any(axAngle != np.pi / 2):
        #     options.append('X||a')
        
        # CS[k] ={'laueGroup':laueGroup, 'axLength':axLength, 'axAngle':axAngle, 'phase':PhaseName[k], 'options':options}

    return CS['id'],CS['name'],CS['point_group'],CS['lattice_constants']

def _get_osc_data(file):
    """ parses tha data stored in an osc file
    to check for integrity, the sript compares whether the x coordinate corresponds with the expected step size

    
    
    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    data - np nd array´
    xstep,ystep - floats 

    """


    # file='map20230512083357665.osc'
    hexArray=['B9','0B','EF','FF','02','00','00','00']
    startBytes= np.array([int(hexNum, 16) for hexNum in list(hexArray)])
        # Open the file and read the header
    with open(file, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=8)
        n = header[6]
        startpos = 0
        bufferLength = 2**20
        
        # Read data from the file
        f.seek(startpos, 0)
        startData = np.fromfile(f, dtype=np.uint8, count=bufferLength)
        startBytes_str=startBytes.astype(str)
        startBytes_str=np.char.zfill(startBytes_str, 4)
        startBytes_str=''.join(startBytes_str)

        startData_str=startData.astype(str)
        startData_str=np.char.zfill(startData_str, 4)
        startData_str=''.join(startData_str)
        
        
        startindex=startData_str.find(startBytes_str)
        startpos=int(startpos+startindex/4)
        # Move to the position after startBytes
        f.seek(startpos+8 , 0)
        # Check for different versions of the .osc file
        dn = np.fromfile(f, dtype=np.uint32, count=1) 
        # if np.round(((dn[0] / 4 - 2) / 10) / n) != 1: # this seems to not necessarily work correctly
            # f.seek(startPos + 8, 1)
        # print(dn)
        # Collect the Xstep and Ystep values
        Xstep = np.fromfile(f, dtype=np.float32, count=1)[0]
        if Xstep == 0:
            Xstep = np.fromfile(f, dtype=np.float32, count=1)[0]
            Ystep = np.fromfile(f, dtype=np.float32, count=1)[0]
        else:
            Ystep = np.fromfile(f, dtype=np.float32, count=1)[0]
        
        # Read the data columns into an array
        position = f.tell()
        for i in range(5, 30):
            data = np.fromfile(f, dtype=np.float32, count=n * i)
            data = data.reshape((n, i))
            data = data.T
            
            # Check if Xstep and Ystep match the data
            if np.isclose(data[3, 1], Xstep, rtol=1e-4) and np.isclose(data[4, 1], 0, rtol=1e-4):
                break
            assert i != 30, "Max number of columns reached, format not handled"
            
            f.seek(position, 0)

    return data, Xstep, Ystep


def _get_vendor_columns(n_cols_file: int) -> Tuple[str, List[str]]:
    """Return the .ang file column names and vendor, determined from the
    header.

    Parameters
    ----------
    header
        List with header lines as individual elements.
    n_cols_file
        Number of file columns.

    Returns
    -------
    vendor
        Determined vendor (``"tsl"``, ``"astar"``, ``"emsoft"`` or
        ``"orix"``).
    column_names
        List of column names.
    """
    # Assume EDAX TSL by default
    vendor = "tsl"

    # Variants of vendor column names encountered in real data sets
    column_names = {
        "tsl": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",  # Image quality from Hough transform
                "ci",  # Confidence index
                "phase_id",
                "detector_signal",
                "fit",  # Pattern fit
                "unknown1",
                "unknown2",
                "unknown3",
                "unknown4",
            ],
            1: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",
                "ci",
                "phase_id",
                "detector_signal",
                "fit",
            ],
        },
        "unknown": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "unknown1",
                "unknown2",
                "phase_id",
            ]
        },
    }
        
    n_variants = len(column_names[vendor])
    n_cols_expected = [len(column_names[vendor][k]) for k in range(n_variants)]
    if n_cols_file not in n_cols_expected:
        warnings.warn(
            f"Number of columns, {n_cols_file}, in the file is not equal to "
            f"the expected number of columns, {n_cols_expected}, for the \n"
            f"assumed vendor '{vendor}'. Will therefore assume the following "
            "columns: euler1, euler2, euler3, x, y, unknown1, unknown2, "
            "phase_id, unknown3, unknown4, etc."
        )
        vendor = "unknown"
        vendor_column_names = column_names[vendor][0]
        n_cols = len(vendor_column_names)
        if n_cols_file > n_cols:
            # Add any extra columns as properties
            for i in range(n_cols_file - n_cols):
                vendor_column_names.append("unknown" + str(i + 3))
    else:
        idx = np.where(np.equal(n_cols_file, n_cols_expected))[0][0]
        vendor_column_names = column_names[vendor][idx]
    return vendor, vendor_column_names

def _test_alternative_pointgroups(point_groups:[str]):
    """ small function to look for other possible namings of the point groups, that seem to not yet be implemented within orix
    
    Needs probably to be extended in the future
    
    """
    for i in range(len(point_groups)):
        if point_groups[i]=='62':
            point_groups[i]='622'
    return point_groups
            

def _is_chemical(phases: [str])-> [str]:
    """ this function shall test, whether the found phases from the binary osc header are actually chemical elements
        it accepts strings that are larger then 4 or strings of lenght 1-2 that are found in the Periodic table of elements
    """

    from periodictable import elements
    elems=[]
    for el in elements:
        elems.append(el.symbol)
    # print(elems)
    elements_input=[]
    pat=re.compile(r'''[A-z]{4,50}(?!\d+)|
                    [A-z]{1,2}(?=\d)|
                    [A-z]{1,2}(?=[A-Z]{1,2})|
                    [A-z]{1,2}(?=[A-Z]{1}[a-z]{1})|
                    [A-z]{1,2}(?=[a-z]{2})''',flags=re.ASCII|re.VERBOSE)
    for phase in phases:
        check=1
        for el_input in re.findall(pat,phase):
            # print(el_input)
            if not el_input in elems:
                check=0
            if len(el_input)>4:
                check=1

        if check==0:
            phases.remove(phase)


    return phases