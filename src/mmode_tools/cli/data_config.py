__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for generating the data configuration files.
"""

import typer
from typing_extensions import Annotated
import toml
from pathlib import Path
import numpy as np
import shutil

from mmode_tools.interferometers import make_radio_array
from mmode_tools.io import get_config_directory,write_data_config

def print_full_line(character='-'):
    """
    Prints a line of a specified character that covers the full width 
    of the terminal window.
    """
    try:
        # Get the terminal size in columns and lines
        columns, lines = shutil.get_terminal_size()
        
        # Print the character repeated to fill the width
        # Subtracting 1 from the width can prevent an extra, unwanted newline 
        # that some terminals automatically add when the very last column is filled.
        print(character * (columns - 1)) 
    except OSError:
        # Fallback for environments where terminal size cannot be determined
        # (e.g., some IDE output consoles)
        print(character * 80) # Default to 80 columns

interferometerPath = get_config_directory(pathName="interferometerPath")
dataPath = get_config_directory(pathName="covTensorPath")

def data_config_main(
    config_files: Annotated[list[str],typer.Option('--config-files','-c',help="NA")] = None,
    data_files: Annotated[list[str],typer.Option('--data-files','-d',help="NA")] = None,
    data_path: Annotated[str,typer.Option('--datapath','-D',help="DataPath")] = dataPath,
    out_path: Annotated[str,typer.Option('--outpath','-O',help="DataPath")] = dataPath,
    config_path: Annotated[str,typer.Option('--config-path','-C',help="DataPath")] = interferometerPath,
    stokes: Annotated[list[str],typer.Option('--stokes','-s',help="NA")] = None,
    lmax_list: Annotated[list[int],typer.Option('--lmax-list','-l',help="NA")] = None,
    freq: Annotated[float,typer.Option('--freq','-f',help="NA")] = 150e6,
    verbose: Annotated[bool,typer.Option("-v",help="Verbose output.")] = False,
    out_config: Annotated[str,typer.Option(help="")] = None):
    
    config_path = Path(config_path)
    data_path = Path(data_path)
    out_path = Path(out_path)

    # Checking that config files are present.
    if config_files is None:
        err = "No .toml config file provided, existing."
        raise ValueError(err)

    # Checking files in list exist.
    for config in config_files:
        if not(Path(config_path/config).exists()):
            err = f"Config file {config} does not exists."
            raise ValueError(err)

    # Checking that data files are present.
    if data_files is None:
        err = "No .hdf5 data file provided, existing."
        raise ValueError(err)
    
    # Checking that data files exist.
    for dataFile in data_files:
        if not(Path(data_path/dataFile).exists()):
            err = f"Data file {dataFile} does not exists."
            raise ValueError(err)

    # Checking that the number of config files matches the number of data files.
    if len(config_files) != len(data_files):
        err = f"length of config_files {len(config_files)} != length of " +\
              f"data_files {len(data_files)}."
        
        raise ValueError(err)


    stokesList = []
    beamFringeFilePaths = []
    interferometerDict = {}
    telescopes = []
    dataFilePaths = []
    lMaxList = []
    for ind,config in enumerate(config_files):
        with open(config_path/config,'r') as f:
            configDict = toml.load(f)
            stokesVec = np.array(configDict["beam-models"]["stokes"])
            freqs = np.array(configDict["beam-models"]["freqs"])
            beamFringes = np.array(configDict["beam-models"]["beamFringeFilePaths"])
            telescope = configDict["params"]["telescope"]
            if np.any(freqs==freq):
                for pol in stokes:
                    boolVec = (freqs==freq)*(stokesVec==pol)
                    dataFilePaths.append(str(data_path/data_files[ind]))
                    interferometerDict[telescope] = make_radio_array(config_path/config)
                    stokesList.append(pol)
                    print(beamFringes[boolVec][0])
                    beamFringeFilePaths.append(str(beamFringes[boolVec][0]))
                    telescopes.append(telescope)
                    lMaxList.append(lmax_list[ind])

    outName = ""
    for telescope in np.unique(telescopes): outName += f"{telescope}_"
    for pol in np.unique(stokesList): outName += f"{pol}_"
    outName += f"{int(freq/1e6)}MHz_"
    outName += "data-config.toml"

    outFilePath = out_path / outName

    if verbose:
        print_full_line(character='=')
        print("Verbose output:")
        print(stokesList)
        print(beamFringeFilePaths)
        print(telescopes)
        print(dataFilePaths)
        print(lMaxList)
        print(outName)
        print(outFilePath)

    write_data_config(outFilePath,dataFilePaths,interferometerDict,telescopes,
                      stokesList,beamFringeFilePaths,freq=freq,dates=None,
                      lMaxList=lMaxList)
    