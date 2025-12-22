__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for cleaning a sky-map.
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

app = typer.Typer()


defaultConfigPath = get_config_directory(pathName="covTensorPath")
defaultInPath = get_config_directory(pathName="dirtyCoeffsPath")
defaultOutPath = get_config_directory(pathName="cleanCoeffsPath")

@app.command()
def make_psf_weights(
    config_file: Annotated[str,
                           typer.Argument(help="Data configuration file, should be .toml.")] = "",
    config_path: Annotated[str,
                      typer.Option("-i",help="Location of the input directory")] = defaultConfigPath,
    inpath: Annotated[str,
                      typer.Option("-i",help="Location of the input directory")] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("-O",help="Location of the output directory")] = defaultOutPath,
    outname: Annotated[str,
                       typer.Option("-o",help="Output name, default is None.")] = None,
    lmax: Annotated[float,typer.Option("-l",
                                       help="Maximum spherical harmonic degree, default = 130.")] = 130,
    n_minor: Annotated[float,typer.Option("-n",
                                          help="Maximum number of minor iterations per major iteration.")] = 1e5,
    n_major: Annotated[float,
                       typer.Option("-N",help="Maximum number of major iterations per major iteration.")] = 10,
    plot: Annotated[bool,
                    typer.Option("-p",help="Plot the primay beam map in RA/DEC.")] = False,
    overwrite: Annotated[bool,
                         typer.Option(help="If True write over old CLEANing results.")] = False,
    verbose: Annotated[bool,
                       typer.Option("-v",help="Print additional information.")] = False
):
    if verbose:
        print(f"config_file: {config_file}")
        print(f"outname: {outname}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
        print(f"Verbose: {verbose}")
        print(f"Plot: {plot}")

        from mmode_tools.io import read_data_config
        configFilePath = config_path + config_file
        mmodeTensor,almTensorList,weights = load_data(configFilePath,
                                              lMax=lmax,freq=freq,
                                              calcWeights=True,
                                              filterParams=None,
                                              uniform=False,flagMmodes=True)

        rhsTensor = calc_psf_weights_tensor(almTensorList,damp=FIcoeffs[0,:,:],
                                            weights=1/sigmaVec)


@app.command()
def clean_map(
    config_file: Annotated[str,
                           typer.Argument(help="Data configuration file, should be .toml.")] = "",
    inpath: Annotated[str,
                      typer.Option("-i",help="Location of the input directory")] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("-O",help="Location of the output directory")] = defaultOutPath,
    outname: Annotated[str,
                       typer.Option("-o",help="Output name, default is None.")] = None,
    lmax: Annotated[float,typer.Option("-l",
                                       help="Maximum spherical harmonic degree, default = 130.")] = 130,
    n_minor: Annotated[float,typer.Option("-n",
                                          help="Maximum number of minor iterations per major iteration.")] = 1e5,
    n_major: Annotated[float,
                       typer.Option("-N",help="Maximum number of major iterations per major iteration.")] = 10,
    plot: Annotated[bool,
                    typer.Option("-p",help="Plot the primay beam map in RA/DEC.")] = False,
    overwrite: Annotated[bool,
                         typer.Option(help="If True write over old CLEANing results.")] = False,
    verbose: Annotated[bool,
                       typer.Option("-v",help="Print additional information.")] = False
):
    
    if verbose:
        print(f"config_file: {config_file}")
        print(f"outname: {outname}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
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
