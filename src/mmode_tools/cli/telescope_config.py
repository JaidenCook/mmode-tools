__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for generating telescope configuration files.
"""

import typer
from typing_extensions import Annotated
import toml
import numpy as np
import os
import importlib.resources as resources

from mmode_tools.interferometers import make_config_file
from mmode_tools.io import get_config_directory


defaultPath = get_config_directory(pathName="interferometerPath")


def telescope_config_main(
    array_layout: Annotated[str,
                            typer.Argument(help="File path to the array layout text file. Should be east, north, height.")] = "",
    telescope: Annotated[str,
                         typer.Option("--name",help="Telescope name.")] = "",
    lat: Annotated[float,
                   typer.Option("-d",help="Latitude of the array. Default is 0.")] = 0,
    lon: Annotated[float,
                   typer.Option("-r",
                                help="Longitude of the array. Default is 0.")] = 0,
    height: Annotated[float,
                   typer.Option("-h",
                                help="Height of the array.")] = 0,
    outpath: Annotated[str,
                       typer.Option("-o",help="Location of the output directory")] = defaultPath,
    override: Annotated[bool,
                       typer.Option("-O",help="Override existing file.")] = False,
    verbose: Annotated[bool,
                       typer.Option("-v",help="Print additional information.")] = False
):
    
    if verbose:
        print(f"Telescope: {telescope}")
        print(f"Array layout file: {array_layout}")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lon}")
        print(f"Your output directory is {outpath}")
        print(f"Override existing file: {override}")
        print(f"Verbose: {verbose}")
    
     # Checking that the array layout file and the beam file path exist.
    if not os.path.exists(array_layout):
        raise FileNotFoundError(f"The specified array layout file does " +\
                                f"not exist: {array_layout}")

    east,north,height = np.loadtxt(array_layout,usecols=(1,2,3),unpack=True)
    antIDs = np.loadtxt(array_layout,usecols=(0),unpack=True,dtype=str)

    if telescope == "":
        Nant = east.shape[0]
        if lat >= 0:
            latStr = f"+{int(lat)}"
        name = f"N{Nant}_{lon}{latStr}"
    else:
        name = telescope

    outName = f"{name}_config.toml"
    outFilePath = outpath + outName
    # Writing the output configuration file.
    make_config_file(outFilePath,arrayLocs=(east,north,height),LAT=lat,LON=lon,
                     HEIGHT=height,antIDs=antIDs,telescope=name,
                     arrayLayout=array_layout,verbose=True)
