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
import importlib.resources as resources

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

defaultInPath = get_config_directory(pathName="covTensorPath")
defaultOutPath = get_config_directory(pathName="dirtyCoeffsPath")

def fit_map_main(
    config_file: Annotated[str,
                         typer.Argument(help="Data configuration file, should be .toml.")] = "",
    lmax: Annotated[float,
                   typer.Option("-l",
                                help="Maximum spherical harmonic degree, default = 130.")] = 130,
    damp: Annotated[float,
                   typer.Option("-d",help="Tikhonov regularisation parameter.")] = 0.01,
    inpath: Annotated[str,
                       typer.Option("-i",help="Location of the input directory")] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("-O",help="Location of the output directory")] = defaultOutPath,
    outname: Annotated[str,
                       typer.Option("-o",help="Output name, default is None.")] = None,
    plot: Annotated[bool,
                       typer.Option("-p",help="Plot the primay beam map in RA/DEC.")] = False,
    filterCond: Annotated[bool,
                         typer.Option("--no-filter",help="If given do not filter the coefficients.")] = True,
    weightsCond: Annotated[bool,
                         typer.Option("--calc-weights",help="If given Calculate the weights.")] = False,
    verbose: Annotated[bool,
                       typer.Option("-v",help="Print additional information.")] = False
):
    
    if verbose:
        print(f"config_file: {config_file}")
        print(f"damp: {damp}")
        print(f"outname: {outname}")
        print(f"filterCond: {filterCond}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
        print(f"lmax: {lmax}")
        print(f"weightsCond: {weightsCond}")
        print(f"Verbose: {verbose}")
        print(f"Plot: {plot}")
    
    if outname is None:
        # If not given then create a name using the configfilepath as a 
        # template.
        outName = os.path.split(config_file)[1].split('.')[0] + ".fits"
        outFilePath = outpath + outName

    #
    with open(inpath+config_file,'r') as f:
        configDict = toml.load(f)
        freq = configDict['params']['freq']

    # Get the filter parameters. This is assumed to be True.
    if filterCond:
        filterParams = make_filter_params(configDict,lMax=lmax)
    else:
        filterParams = None

    # Load the data in the mmode tensor format. Additionally load the weights
    # and the beam fringe coefficients for each of the baselines
    mmodeTensor,almTensorList,weights = load_data(inpath+config_file,
                                                  lMax=lmax,freq=freq,
                                                  calcWeights=weightsCond,
                                                  filterParams=filterParams,
                                                  verbose=verbose)

    if verbose:
        verbosity = 10
        print("Performing the inversion.")
    else:
        verbosity = 0
    # Perform the inversion, return the CAR map and the coefficients.
    skyMap,skyCo = data2map(mmodeTensor,almTensorList,weights,
                            invert=invert_tikh_multi_assym,lMax=lmax,
                            damp=damp,verbosity=verbosity,returnCoeffs=True,
                            damp_alpha=0)
    

    # Saving the output map.
    map2fits(skyMap.real,freq,outFilePath,skyCoeffs=skyCo)

    if verbose:
        print(f"Map saved to {outFilePath}...")
