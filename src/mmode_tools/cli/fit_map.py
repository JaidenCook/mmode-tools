__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for fitting sky maps to data and generating dirty sky
coefficients.
"""

import typer
from typing_extensions import Annotated
import toml
import numpy as np
import os
import shutil
import importlib.resources as resources
from warnings import warn

from mmode_tools.inversion import invert_tikh_multi_assym
from mmode_tools.io import get_config_directory
from mmode_tools.io import map2fits
from mmode_tools.utils import data2map,load_data

def make_filter_params(configDict,lMax=None):
    """
    Read the config file and and generate the filter parameter dictionary.
    """
    if isinstance(configDict,str):
        with open(configDict,'r') as f:
            configDict = toml.load(f)
    elif isinstance(configDict,dict):
        pass

    print(configDict["params"]["telescope_config_files"])

    telescopes = configDict['params']['telescopes']
    lMaxList = configDict['params']['lMaxList']

    filterParams = {}
    for i,telescope in enumerate(telescopes):
        mMax = lMaxList[i]

        if lMax is not None:
            if mMax > lMax:
                mMax = lMax

        filterParams[telescope] = {
            "lcut" : mMax-10,
            "lwin" : 10,
            "lmax" : mMax,
            "telescope" : telescope
        }

    return filterParams

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

defaultInPath = get_config_directory(pathName="covTensorPath")
defaultOutPath = get_config_directory(pathName="dirtyCoeffsPath")

from typing import List, Optional

def fit_map_main(
    config_file: Annotated[str,
                         typer.Argument(help="Data configuration file, should be .toml.")] = "",
    lmax: Annotated[Optional[List[int]],
                   typer.Option("--lmax","-l",
                                help="Maximum spherical harmonic degree, default = 130.")] = [130],
    damp: Annotated[float,
                   typer.Option("--damp","-d",help="Regularisation parameter.")] = 0.01,
    inpath: Annotated[str,
                       typer.Option("--inpath","-i",help="Location of the input directory")] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("--outpath","-O",help="Location of the output directory")] = defaultOutPath,
    outname: Annotated[str,
                       typer.Option("--outname","-o",help="Output name, default is None.")] = None,
    plot: Annotated[bool,
                       typer.Option("--plot","-p",help="Plot the primay beam map in RA/DEC.")] = False,
    calc_fisher: Annotated[bool,
                       typer.Option("--calc-fisher","-F",help="If True calc Fisher information, overrides damp.")] = False,
    filterCond: Annotated[bool,
                         typer.Option("--filter",help="If given do not filter the coefficients.")] = False,
    flag_mmodes: Annotated[bool,
                         typer.Option("--flag-mmodes",help="If given flag-mmodes larger than lmax.")] = False,
    weightsCond: Annotated[bool,
                         typer.Option("--calc-weights",help="If given Calculate the weights.")] = False,
    verbose: Annotated[bool,
                       typer.Option("-v",help="Print additional information.")] = False
):
    

    #
    with open(inpath+config_file,'r') as f:
        configDict = toml.load(f)
        freq = configDict['params']['freq']
        lMaxVec = np.array(configDict['params']['lMaxList'])
        
        if np.log10(freq) > 6:
            # Frequency is in Hz, most functions accept MHz. Stupid fix 
            # Could use astropy units to solve this in the future.
            print('Frequency in Hz, converting to MHz.')
            freq /= 1e6

    # Get the filter parameters. This is assumed to be True.
    if filterCond:
        filterParams = make_filter_params(configDict,lMax=lmax)
    else:
        filterParams = None

    if len(lmax) > 1:
        lMax = np.array(lmax).max()
        if len(lmax) == lMaxVec.size:
            lMaxVec = np.array(lmax)
    else:
        lMax = lmax[0]
    
    if lMax > lMaxVec.max():
        lMax = lMaxVec.max()
    elif lMax < lMaxVec.max():
        lMaxVec[lMaxVec > lMax] = lMax

    if outname is None:
        # If not given then create a name using the configfilepath as a 
        # template.
        prefix = os.path.split(config_file)[1].split('.')[0]
        outName = prefix +f"_lmax{lMax}" + "_dirty-map.fits"
        outFilePath = outpath + outName
    else:
        outName = outname

    if verbose:
        print('Input arguments and optional arguments:')
        print_full_line(character='=')
        print(f"config_file: {config_file}")
        print(f"damp: {damp}")
        print(f"outname: {outName}")
        print(f"filterCond: {filterCond}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
        print(f"lmax: {lmax}")
        print(f"lMaxVec: {lMaxVec}")
        print(f"freq: {freq} [MHz]")
        print(f"weightsCond: {weightsCond}")
        print(f"Verbose: {verbose}")
        print(f"Plot: {plot}")
        print_full_line(character='=')


    print("Loading the mmodeTensor, and beam fringe coefficients.")    
    print_full_line(character='=')

    # Load the data in the mmode tensor format. Additionally load the weights
    # and the beam fringe coefficients for each of the baselines
    mmodeTensor,almTensorList,weights = load_data(inpath+config_file,
                                                  lMax=lMax,freq=freq,
                                                  calcWeights=weightsCond,
                                                  filterParams=filterParams,
                                                  verbose=verbose,
                                                  flagMmodes=flag_mmodes)

    #
    if lMaxVec is not None:
        if lMaxVec.size != len(almTensorList):
            warn(f"Number of Lmax values {lMaxVec.size} should equal number of " +
                f"elements in the almTensorList {len(almTensorList)}. " +
                "Setting lMaxVec to None.")
            lMaxVec = None

    #
    if calc_fisher:
        print("Calculating the Fisher Information (FI) for regularisation...")
        print("Overriding damp value with FI...")
        print_full_line(character='=')

        from mmode_tools.inversion import calc_fisher_coeffs
        if weightsCond == False:
            # If no noise condition assume unity noise weights for all
            # instruments.
            noiseVec = np.ones(mmodeTensor.shape[0])
        else:
            noiseVec = 1/weights
            noiseVec[weights==0] = 0
        damp = calc_fisher_coeffs(almTensorList,noiseVec,lMax=lMax,
                                  lMaxVec=lMaxVec)
        print(damp.shape)
    #
    if verbose:
        verbosity = 10
        print("Performing the inversion.")
        print_full_line(character='=')
    else:
        verbosity = 0

    # Perform the inversion, return the CAR map and the coefficients.
    skyMap,skyCo = data2map(mmodeTensor,almTensorList,weights,
                            invert=invert_tikh_multi_assym,lMax=lMax,
                            lMaxVec=lMaxVec,damp=damp,verbosity=verbosity,
                            returnCoeffs=True,damp_alpha=0)
    
    # Saving the output map.
    map2fits(skyMap.real,freq,outFilePath,skyCoeffs=skyCo,damp=damp)

    if verbose:
        print(f"Map saved to {outFilePath}...")
