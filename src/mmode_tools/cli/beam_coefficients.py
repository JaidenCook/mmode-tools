__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for generating the beam fringe spherical harmonic
coefficients for a given telescope configuration and primary beam model.
"""

import typer
from typing_extensions import Annotated
import toml
import numpy as np
import os
import importlib.resources as resources
from mmode_tools.beam import MWA_dipolebeam,FITS2beam,bline2alm_h5py
from mmode_tools.interferometers import MWAPH2array,EDA2array,RadioArray,make_radio_array
from mmode_tools.io import get_config_directory


def memCalc(Nant,lMax,dt):
    nbytes = np.dtype(dt).itemsize
    return 2*nbytes*Nant*(Nant-1)*(lMax)**2 / 1024**3

#
interferometerPath = get_config_directory(pathName="interferometerPath")
beamPath = get_config_directory(pathName="beamPath")
defaultOutPath = get_config_directory(pathName="beamFringePath")

def beam_coefficients_main(
    config_file: Annotated[str,
                         typer.Argument(help="Telescope configuration file name.")] = "",
    beam_file: Annotated[str,
                         typer.Option("-b","--beam-file",help="File path to the beam coefficients text file.")] = "",
    pol: Annotated[str,
                   typer.Option("-P","--pol",help="Instrument polarisation, default is XX.")] = "XX",
    freq: Annotated[float,
                   typer.Option("-f","--freq",help="Frequency channel in Hz, default is 160MHz.")] = 150e6,
    lmax: Annotated[int,
                   typer.Option("-l","--lmax",
                                help="Maximum spherical harmonic degree, default = 130.")] = 130,
    outpath: Annotated[str,
                       typer.Option("-O","--outpath",help="Location of the output directory")] = defaultOutPath,
    outname: Annotated[str,
                       typer.Option("-o","--outname",help="Location of the output directory")] = None,
    plot: Annotated[bool,
                       typer.Option("-p","--plot",help="Plot the primay beam map in RA/DEC.")] = False,
    chunks: Annotated[bool,
                       typer.Option("-c",help="If True chunk the coefficients..")] = False,
    compression: Annotated[str,
                         typer.Option("--compression",help="Compression to use, only works if chunks set to True..")] = "lzf",
    verbose: Annotated[bool,
                       typer.Option("-v","--verbose",help="Print additional information.")] = False
):
    
    if verbose:
        print(f"config_file: {config_file}")
        print(f"beam_file: {beam_file}")
        print(f"pol: {pol}")
        print(f"freq: {freq}")
        print(f"Your output directory is {outpath}")
        print(f"lmax: {lmax}")
        print(f"Verbose: {verbose}")
        print(f"Plot: {plot}")
        print(f"Chunks: {chunks}")
        print(f"Compression: {compression}")
    
    # Check that the output that exists if not make it.
    if not os.path.exists(outpath):
        raise FileNotFoundError(f"The {outpath} does not exist.")

    if not os.path.exists(interferometerPath+config_file):
        raise FileNotFoundError(f"The specified configuration file does not " +\
                                f"exist: {config_file}")


    with open(interferometerPath+config_file, 'r') as f:
        config = toml.load(f)
        arrayLat = config['location']['lat']
        telescope = config['params']['telescope']
        override = config['params']['override']

    # Making the radio array object.
    if (telescope == "MWA") and override:
        Array = MWAPH2array
    elif (telescope == "EDA2") and override:
        Array = EDA2array
    else:
        Array = make_radio_array(interferometerPath+config_file,lat=arrayLat)

    if outname == None:
        outName = "beam_fringe_coeffs"
        if telescope:
            outName += f"-{telescope}"
        if freq:
            outName += f"-{int(freq/1e6)}MHz"
        if pol:
            outName += f"-{pol}"
        if lmax:
            outName += f"-lMax{lmax}"
        if chunks:
            outName += f"-chunked"
            if compression:
                outName += f"-{compression}.hdf5"
        else:
            outName += f".hdf5"
        
    else:
        outName = outname

    outFilePath = outpath+outName
    #
    Ncells = int(2*lmax+2)

    print(f"Telescope : {telescope}")
    print(f"Latitude : {arrayLat:5.3f} [deg]")
    print(f"Instrument polarisation : {pol}")

    blines,antPairs = RadioArray.get_baselines(Array,calcAutos=False)
    
    if telescope == 'ONSALA':
        # Fixing a conjugate error in the simulated data.
        blines *= -1

    memMax = memCalc(Array.Nant,lmax,np.complex64)
    print(f"Will need, {memMax:5.3f}GB of RAM for the m-mode coefficients.")

    # Loading in the beam map.
    if telescope == "MWA":
        # For the MWA we can generate the beam map using the mwa_hyperbeam
        # package.
        if pol == "XX" or pol == "X" or pol == "xx" or pol == "x":
            pol = "X"
        elif pol == "YY" or pol == "Y" or pol == "yy" or pol == "y":
            pol = "Y"
        beamMap = MWA_dipolebeam(freq,np.radians(arrayLat),Ncells,pol=pol)
    else:
        if beam_file == "" or not os.path.exists(beam_file):
            from mmode_tools.examples import load_default_beam_model
            print("Provided beam file does not exist, " \
            "loading default beam model.")

            beamMap = load_default_beam_model(LAT=np.radians(arrayLat),
                                              pol=pol,lMax=lmax)
        else:
            # Load in the primary beam map form the fits file.
            print(f"Ncells = {Ncells}")
            beamMap = FITS2beam(beam_file,np.radians(arrayLat),lmax)


    # If True plot the primary beam map.
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(beamMap,origin='lower')
        plt.show()
    
    # Generating the alm beam coefficients.
    bline2alm_h5py(blines,antPairs,beamMap,freq,np.radians(arrayLat),lmax,
                   outFilePath,chunks=chunks,compression=compression)