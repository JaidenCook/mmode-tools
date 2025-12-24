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
from typing import List, Optional
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

# Putting all the help lines here to make things more concise.
helpList = ["Data configuration file, should be .toml.",
            "Maximum spherical harmonic degree, default = 130.",
            "Regularisation parameter.","Location of the input directory",
            "Location of the output directory","Output name, default is None.",
            "Plot the primay beam map in RA/DEC.",
            "If True calc Fisher information, overrides damp.",
            "If given do not filter the coefficients.",
            "If given flag-mmodes larger than lmax.",
            "If given Calculate the weights.","Print additional information."]

def fit_map_main(
    config_file: Annotated[str,typer.Argument(help=helpList[0])] = "",
    lmax: Annotated[Optional[List[int]],typer.Option("--lmax","-l",help=helpList[1])] = [130],
    damp: Annotated[float,typer.Option("--damp","-d",help=helpList[2])] = 0.01,
    inpath: Annotated[str,typer.Option("--inpath","-i",help=helpList[3])] = defaultInPath,
    outpath: Annotated[str,typer.Option("--outpath","-O",help=helpList[4])] = defaultOutPath,
    outname: Annotated[str,typer.Option("--outname","-o",help=helpList[5])] = None,
    plot: Annotated[bool,typer.Option("--plot","-p",help=helpList[6])] = False,
    calc_fisher: Annotated[bool,typer.Option("--calc-fisher","-F",help=helpList[7])] = False,
    filterCond: Annotated[bool,typer.Option("--filter",help=helpList[8])] = False,
    flag_mmodes: Annotated[bool,typer.Option("--flag-mmodes",help=helpList[9])] = False,
    weightsCond: Annotated[bool,typer.Option("--calc-weights",help=helpList[10])] = False,
    verbose: Annotated[bool,typer.Option("-v",help=helpList[11])] = False
):
    # Loading in the some of the important meta data.
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

    print("Loading the mmodeTensor, and beam fringe coefficients.")    
    print_full_line(character='=')

    # Load the data in the mmode tensor format. Additionally load the weights
    # and the beam fringe coefficients for each of the baselines.
    # When loading the data we want a large lmax, for calculating the expected
    # noise on the mmodes. That's because we use the noise dominated modes to 
    # estimate the noise amplitude from the difference visibilities. If lmax
    # is too low then longer baselines don't get accurate noise estimates.
    mmodeTensor,almTensorList,weights = load_data(inpath+config_file,
                                                  lMax=int(lMaxVec.max()),
                                                  freq=freq,
                                                  calcWeights=weightsCond,
                                                  filterParams=filterParams,
                                                  verbose=verbose,
                                                  flagMmodes=flag_mmodes)

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
    else:
        outName = outname
    outFilePath = outpath + outName

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

    # Check lMaxVec has the right size.
    if lMaxVec is not None:
        if lMaxVec.size != len(almTensorList):
            warn(f"Number of Lmax values {lMaxVec.size} should equal number of " +
                f"elements in the almTensorList {len(almTensorList)}. " +
                "Setting lMaxVec to None.")
            lMaxVec = None

    # If True calculate the Fisher information for regularisation.
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
                                  lMaxVec=lMaxVec,absOffset=1e-1)
        # Only need the positive m-mode regularisation parameters.
        damp = damp[0,:lMax+1,:lMax+1]
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
    
    # Saving the output map, sky coefficients and regularisation parameter.
    map2fits(skyMap.real,freq,outFilePath,skyCoeffs=skyCo,damp=damp)

    if verbose:
        print(f"Map saved to {outFilePath}...")
