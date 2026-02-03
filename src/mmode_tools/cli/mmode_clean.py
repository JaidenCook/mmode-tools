__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

"""
Command line tool for cleaning a sky-map.
"""

import typer
import shutil
from typing import Tuple,Optional,List
from typing_extensions import Annotated
import toml
import numpy as np
import os
import importlib.resources as resources
import h5py as h5
from pathlib import Path
from mmode_tools.skymap import SkyMap


from mmode_tools.inversion import invert_tikh_multi_assym
from mmode_tools.io import get_config_directory
from mmode_tools.io import map2fits
from mmode_tools.utils import data2map,load_data
from mmode_tools.io import fits2skyCoeffs
from mmode_tools.clean import calc_psf_weights_tensor
from mmode_tools.io import read_data_config
from mmode_tools.clean import calc_clean_mask,make_mask_box

def calc_peak_max_vec(dirtySkyMap,psfWeightsTensor,lmax,stride=10):
    """calc_peak_max_vec _summary_

    Parameters
    ----------
    dirtySkyMap : _type_
        _description_
    psfWeightsTensor : _type_
        _description_
    lmax : _type_
        _description_
    stride : int, optional
        _description_, by default 10

    Returns
    -------
    _type_
        _description_
    """
    
    latVec = dirtySkyMap.colat[::stride]-90
    dirtyPeakMaxVec = np.zeros(dirtySkyMap.colat[::stride].size)
    dirtyMonopoleVec = np.zeros(dirtySkyMap.colat[::stride].size,
                                dtype=np.complex64)
    from tqdm import tqdm
    from mmode_tools.skymap import convolve_model_map,calc_analytic_ps
    for ind,colat in enumerate(tqdm(dirtySkyMap.colat[::stride])):

        amp = 1/np.cos(np.radians(colat-90))
        almPs = calc_analytic_ps(colat,180,amp,lMax=lmax)

        modelPointMap = SkyMap(coeffs=almPs)
        dirtyModelPointMap = convolve_model_map(modelPointMap,psfWeightsTensor,
                                                lMax=lmax)

        dirtyPeakMaxVec[ind] = dirtyModelPointMap.skyMap.max()
        dirtyMonopoleVec[ind] = dirtyModelPointMap.coeffs[0,0,0]
    
    return latVec,dirtyPeakMaxVec,dirtyMonopoleVec

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
        # that some terminals automatically add when the very last column is 
        # filled.
        print(character * (columns - 1)) 
    except OSError:
        # Fallback for environments where terminal size cannot be determined
        # (e.g., some IDE output consoles)
        print(character * 80) # Default to 80 columns

app = typer.Typer()

defaultConfigPath = get_config_directory(pathName="covTensorPath")
defaultInPath = get_config_directory(pathName="dirtyCoeffsPath")
defaultOutPath = get_config_directory(pathName="cleanCoeffsPath")

helpListCmd1 = ["Data configuration file, should be .toml.",
                "Dirty map fits file name.",
                "Location of the config file directory",
                "Location of the input directory",
                "Location of the output directory",
                "Output name, default is None.",
                "Maximum spherical harmonic degree, default = 130.",
                "Weights condition.",
                "If True calculate the peak max PSF as a function of latitude.",
                "Print additional information."]
@app.command()
def make_psf_weights(
    config_file: Annotated[str,typer.Argument(help=helpListCmd1[0])] = "",
    dirty_map: Annotated[str,typer.Argument(help=helpListCmd1[1])] = "",
    config_path: Annotated[str,typer.Option("-c",help=helpListCmd1[2])] = defaultConfigPath,
    inpath: Annotated[str,typer.Option("-i",help=helpListCmd1[3])] = defaultInPath,
    outpath: Annotated[str,typer.Option("-O",help=helpListCmd1[4])] = defaultOutPath,
    outname: Annotated[str,typer.Option("-o",help=helpListCmd1[5])] = None,
    lmax: Annotated[int,typer.Option("-l",help=helpListCmd1[6])] = 130,
    weightsCond: Annotated[bool,typer.Option("--calc-weights",help=helpListCmd1[7])] = False,
    calc_peak_vec: Annotated[bool,typer.Option("--calc-peak-vec",help=helpListCmd1[8])] = False,
    verbose: Annotated[bool,typer.Option("-v",help=helpListCmd1[9])] = False
):
    """make_psf_weights Makes the psfWeightsTensors required for making PSF maps.
    This follows the same method as Eastwood et al. 2018.

    Parameters
    ----------
    config_file : Annotated[str,typer.Argument, optional
        _description_, by default helpListCmd1[0])]=""
    dirty_map : Annotated[str,typer.Argument, optional
        _description_, by default helpListCmd1[1])]=""
    config_path : Annotated[str,typer.Option, optional
        _description_, by default helpListCmd1[2])]=defaultConfigPath
    inpath : Annotated[str,typer.Option, optional
        _description_, by default helpListCmd1[3])]=defaultInPath
    outpath : Annotated[str,typer.Option, optional
        _description_, by default helpListCmd1[4])]=defaultOutPath
    outname : Annotated[str,typer.Option, optional
        _description_, by default helpListCmd1[5])]=None
    lmax : Annotated[int,typer.Option, optional
        _description_, by default helpListCmd1[6])]=130
    weightsCond : Annotated[bool,typer.Option, optional
        _description_, by default helpListCmd1[7])]=False
    overwrite : Annotated[bool,typer.Option, optional
        _description_, by default helpListCmd1[8])]=False
    verbose : Annotated[bool,typer.Option, optional
        _description_, by default helpListCmd1[9])]=False

    Raises
    ------
    ValueError
        _description_
    """
    if isinstance(inpath,str):
        inpath = Path(inpath)
    
    if isinstance(outpath,str):
        outpath = Path(outpath)
    
    if isinstance(config_path,str):
        config_path = Path(config_path)

    # Config file and dirty map filepaths.
    dirtyMapFilePath = inpath / dirty_map
    configFilePath = config_path / config_file

    # Loading in the some of the important meta data.
    with open(configFilePath,'r') as f:
        configDict = toml.load(f)
        freq = configDict['params']['freq']
        lMaxVec = np.array(configDict['params']['lMaxList'])
        
        if np.log10(freq) > 6:
            # Frequency is in Hz, most functions accept MHz. Stupid fix 
            # Could use astropy units to solve this in the future.
            print('Frequency in Hz, converting to MHz.')
            freq /= 1e6

    #    
    if lmax > lMaxVec.max():
        lmax = lMaxVec.max()
    elif lmax < lMaxVec.max():
        lMaxVec[lMaxVec > lmax] = lmax

    if outname is None:
        # If not given then create a name using the configfilepath as a 
        # template.
        #prefix = os.path.split(config_file)[1].split('.')[0].split('config')[0]
        prefix = os.path.split(config_file)[1].split('.')[0]
        outName = prefix +f"lmax{lmax}" + "_clean-components.hdf5"
    else:
        outName = outname
    outFilePath = outpath / outName

    if verbose:
        print(f"config_file: {config_file}")
        print(f"dirty map file: {dirty_map}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
        print(f"output file name = {outName}")
        print(f"Verbose: {verbose}")
        print_full_line()

    # Initialising the dirty map into a SkyMap object.
    dirtySkyMap = SkyMap(dirtyMapFilePath)
    
    # Getting the damping/regularisation parameter(s).
    damp = dirtySkyMap.weights # Weights are the SH damping coefficients.
    if isinstance(damp,np.ndarray):
        if damp.ndim == 3:
            # Case where there are reg params for each alm.
            # We only care about the positive m-mode values.
            damp = damp[0,:,:]

    # Loading in the data.
    _,almTensorList,weights = load_data(configFilePath,lMax=lmax,freq=freq,
                                        calcWeights=weightsCond,
                                        filterParams=None,
                                        uniform=False,flagMmodes=True)

    if weights is None:
        # If weights are none then assign ones to weights.
        Nbase = sum([alm.shape[0] for alm in almTensorList])
        weights = np.ones(Nbase)

    print_full_line()
    # Calculating the PSFweights Tensor.
    psfWeightsTensor = calc_psf_weights_tensor(almTensorList,damp=damp,
                                               weights=weights)
    print_full_line()
    if calc_peak_vec:
        print("Calculating the dirty peak max vector as a function of latitude.")
        latVec,dirtyPeakVec,dirtyMonopoleVec = calc_peak_max_vec(dirtySkyMap,
                                                                 psfWeightsTensor,
                                                                 lmax,stride=10)
    else:
        latVec,dirtyPeakVec,dirtyMonopoleVec = None,None,None

    # Saving the psfWeightsTesnor to an ouptut hdf5 file.
    if outFilePath.exists():
        # Check if there is an existing file.
        with h5.File(outFilePath,'a') as hf:
            hf.attrs['dirtyFilePath'] = str(dirtyMapFilePath)
            hf.attrs['configFilePath'] = str(configFilePath)
            try:
                group = hf.create_group("psfWeightsTensor")
            except ValueError:
                print('Group data already exists.')
                dset = hf['psfWeightsTensor/weightsTensor']

                if dset.shape == psfWeightsTensor.shape:
                    # If the shapes are equal we can assign the new psf weight 
                    # tensor. Otherwise we raise a value error.
                    dset[...] = psfWeightsTensor
                else:
                    err = f'dset.shape {dset.shape} != psfWeightsTensor.shape'+\
                            f' {psfWeightsTensor.shape}.'
                    raise ValueError(err)
                
                if latVec is not None:
                    # overwriting the dirty peak vec.
                    group.attrs['latVec'][...] = latVec
                    group.attrs['dirtyPeakVec'][...] = dirtyPeakVec
                    group.attrs['dirtyMonopoleVec'][...] = dirtyMonopoleVec
    else:
        # If the file does not exist we create one.
        with h5.File(outFilePath,'w') as hf:
            hf.attrs['dirtyFilePath'] = str(dirtyMapFilePath)
            hf.attrs['configFilePath'] = str(configFilePath)

            # Creating the weights tensor dataset and group.
            group = hf.create_group("psfWeightsTensor")
            group.create_dataset("weightsTensor",data=psfWeightsTensor)

            if latVec is not None:
                # Assigning the dirty peak vec.
                group.attrs['latVec'] = latVec
                group.attrs['dirtyPeakVec'] = dirtyPeakVec
                group.attrs['dirtyMonopoleVec'] = dirtyMonopoleVec
    #
    print_full_line()
    print(f"File saved to {outFilePath}")


@app.command()
def make_mask_list(
    dirty_map: Annotated[str,typer.Argument(help="Dirty Map")] = "",
    #mask_params: Annotated[List[Tuple[str,float,float,float]],typer.Argument(help="(Name,RA,DEC,size) #[deg]")] = "",
    mask_params: Annotated[List[str],typer.Argument(help="(Name,RA,DEC,size) #[deg]")] = None,
    inpath: Annotated[str,
                      typer.Option("-i",help="inpath")] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("-O",help="outpath")] = defaultOutPath,
    outname: Annotated[str,typer.Option("-o",help="outname")] = None,
    overwrite: Annotated[bool,typer.Option("--no-overwrite",help="Overwrite file condition, default is True.")] = True,
    
):
    from pyshtools import SHCoeffs

    dirtyMapFilePath = inpath / dirty_map
    skyCo,_ = fits2skyCoeffs(dirtyMapFilePath,readRegParams=True)

    coeffsObj = SHCoeffs.from_array(skyCo,normalization='ortho',csphase=-1)
    coeffsObjExp = coeffsObj.expand(grid='DH2',backend='ducc')
    RAVec = coeffsObjExp.lons()
    DECVec = coeffsObjExp.lats()
    RAVecNew = np.roll(np.copy(RAVec),int(RAVec.size/2))
    RAgrid,DECgrid = np.meshgrid(RAVecNew,DECVec[::-1])

    #
    maskParamsZip = []
    for params in mask_params:
        paramsList = params.split(',')
        paramsTup = (paramsList[0],float(paramsList[1]),float(paramsList[2]),
                     float(paramsList[3]))
        maskParamsZip.append(paramsTup)
    
    maskList = make_mask_box(maskParamsZip,RAgrid,DECgrid)

    #
    if outname is None:
        # If not given then create a name using the configfilepath as a 
        # template.
        outName = dirty_map.split("_dirty-map.fits")[0] + "_mask-List.toml"
        outFilePath = outpath + outName

    # Outputting the configuration file to a .toml file.
    if os.path.exists(outFilePath):
        if overwrite:
            maskDict = {"maskList_0": maskList}
            with open(outFilePath,"w") as f:
                toml.dump(maskDict, f)
        else:
            # If overwrite is false we can append more mask liks to the old one.
            # Bit convoluted. 
            with open(outFilePath,"r") as f:
                maskDictInitial = toml.load(f)
                nkeys = len(maskDictInitial.keys())
                maskDict = {f"maskList_{nkeys}": maskList}
                maskDict = maskDictInitial | maskDict

            with open(outFilePath,"w") as f:
                toml.dump(maskDict, f)
    else:
        maskDict = {"maskList_0": maskList}
        with open(outFilePath,"w") as f:
            toml.dump(maskDict, f)

    print(f"Mask list file written to {outFilePath}...")



helpListCmd2 = ["Data configuration file, should be .toml.",
                "Dirty map fits file name.",
                "Location of the config file directory",
                "Location of the input directory",
                "Location of the output directory",
                "Output name, default is None.",
                "Maximum spherical harmonic degree, default = 130.",
                "Number of major iteration, default=10.",
                "Number of minor iterations, default=1e5",
                "Loop gain per minor iteration.",
                "Default (-90,90), provide lower and upper bound to declination for clean mask",
                "Galactic latitude threshold.",
                "Flip the GP threshold conditions.",
                "Filepath to the masklist for creating the clean mask.",
                "SNR threshold for peak detection.",
                "Std threshold for CLEAN.",
                "If True write over old CLEANing results.",
                "Print additional information."]

@app.command()
def clean_map(
    config_file: Annotated[str,typer.Argument(help=helpListCmd2[0])] = "",
    dirty_map: Annotated[str,typer.Argument(help=helpListCmd2[1])] = "",
    config_path: Annotated[str,
                           typer.Option("-i",help=helpListCmd2[2])] = defaultConfigPath,
    inpath: Annotated[str,
                      typer.Option("-i",help=helpListCmd2[3])] = defaultInPath,
    outpath: Annotated[str,
                       typer.Option("-O",help=helpListCmd2[4])] = defaultOutPath,
    outname: Annotated[str,typer.Option("-o",help=helpListCmd2[5])] = None,
    lmax: Annotated[int,typer.Option("-l",help=helpListCmd2[6])] = 130,
    n_major: Annotated[int,typer.Option("-N",help=helpListCmd2[7])] = 10,
    n_minor: Annotated[int,typer.Option("-n",help=helpListCmd2[8])] = 1e5,
    loop_gain: Annotated[float,
                         typer.Option("--loop-gain",help=helpListCmd2[9])] = 0.01,
    dec_thresh: Annotated[Optional[Tuple[float,float]],
                          typer.Option("--dec-limits",
                                       help=helpListCmd2[10])] = (-90,90),
    gp_thresh: Annotated[float,typer.Option("--gp-thresh",help=helpListCmd2[11])] = 0,
    gp_flip: Annotated[bool,typer.Option("--gp-flip",help=helpListCmd2[12])] = False,
    mask_list: Annotated[str,typer.Option("--mask_list",help=helpListCmd2[13])] = None,
    thresh: Annotated[Optional[List[int]],
                      typer.Option("--thresh","-t",help=helpListCmd2[14])] = [7],
    sig_thresh: Annotated[Optional[List[int]],
                          typer.Option("--sig-thresh","-s",help=helpListCmd2[15])] = [1],
    overwrite: Annotated[bool,typer.Option(help=helpListCmd2[16])] = False,
    verbose: Annotated[bool,typer.Option("-v",help=helpListCmd2[17])] = False
):
    from mmode_tools.clean import major_iteration
    from pyshtools import SHCoeffs

    dirtyMapFilePath = inpath + dirty_map
    configFilePath = config_path + config_file

    if verbose:
        print(f"config_file: {config_file}")
        print(f"outname: {outname}")
        print(f"Your input directory is {inpath}")
        print(f"Your output directory is {outpath}")
        print(f"Verbose: {verbose}")
    
    if outname is None:
        # If not given then create a name using the configfilepath as a 
        # template.
        outName = os.path.split(config_file)[1].split('.')[0] + ".fits"
        outFilePath = outpath + outName

    #
    with open(inpath+config_file,'r') as f:
        configDict = toml.load(f)
        freq = configDict['params']['freq']
        lMaxVec = np.array(configDict['params']['lMaxList']).astype(int)

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

    # Getting the damping/regularisation parameter(s).
    skyCo,damp = fits2skyCoeffs(dirtyMapFilePath,readRegParams=True)
    if isinstance(damp,np.ndarray):
        if damp.ndim == 3:
            # Case where there are reg params for each alm.
            damp = damp[0,:,:]

    # Load the data in the mmode tensor format. Additionally load the weights
    # and the beam fringe coefficients for each of the baselines
    mmodeTensor,almTensorList,_ = load_data(configFilePath,lMax=lmax,
                                                  freq=freq,
                                                  calcWeights=False,
                                                  filterParams=None,
                                                  uniform=False,flagMmodes=True)

    coeffsObj = SHCoeffs.from_array(skyCo,normalization='ortho',csphase=-1)
    coeffsObjExp = coeffsObj.expand(grid='DH2',backend='ducc')
    dirtyMap = coeffsObjExp.data.real
    RAVec = coeffsObjExp.lons()
    DECVec = coeffsObjExp.lats()
    RAVecNew = np.roll(np.copy(RAVec),int(RAVec.size/2))


    RAgrid,DECgrid = np.meshgrid(RAVecNew,DECVec[::-1])

    modelMap = np.zeros(dirtyMap.shape)
    
    prefix = os.path.split(config_file)[1].split('.')[0]
    cleanComponentsFileName = prefix +f"_lmax{lMax}" + "_clean-components.hdf5"
    psfWeightsFilePath = defaultInPath + cleanComponentsFileName
    with h5.File(psfWeightsFilePath,'r') as hf:
        psfWeightsTensor = hf['psfTensors/FI_tensor'][:]
        
    psfCoeffsCube = np.zeros((dirtyMap.shape[0],lMax+1,lMax+1),
                             dtype=np.complex64)

    if len(thresh) == 1:
        # Create new list with the n_major elements of equal value.
        thresh = thresh * n_major
    elif len(thresh) != n_major:
        # If thresh list has more than 1 element but fewer than n_major,
        # make new list from max to min of given list. This could be a case
        # where the min and max are procided by the user.
        threshMin = min(thresh)
        threshMax = max(thresh)
        thresh = np.linspace(threshMax,threshMin,n_major).astype(int)

    if len(sig_thresh) == 1:
        # Create new list with the n_major elements of equal value.
        sig_thresh = sig_thresh * n_major
    elif len(sig_thresh) != n_major:
        # If thresh list has more than 1 element but fewer than n_major,
        # make new list from max to min of given list. This could be a case
        # where the min and max are procided by the user.
        threshMin = min(sig_thresh)
        threshMax = max(sig_thresh)
        sig_thresh = np.linspace(threshMax,threshMin,n_major).astype(int)
    
    # If given load list.
    if mask_list is not None:
        if os.path.exists(mask_list):
            with open(inpath+config_file,'r') as f:
                configDict = toml.load(f)
                maskList = configDict['maskLists']['maskList']
    # Calculating the clean mask.
    cleanMask = calc_clean_mask(skyCo,DECthresh=dec_thresh,maskList=maskList,
                                plotCond=False,GPthresh=gp_thresh,
                                GPthreshFlip=gp_flip)

    # Major iterations.
    for majorIter in range(n_major):
        loopCond = major_iteration(mmodeTensor,almTensorList,dirtyMap.real,
                                   modelMap,psfCoeffsCube,psfWeightsTensor,
                                   DECgrid,skyCo,Nminor=n_minor,plotCond=True,
                                   thresh=thresh[majorIter],lMax=lMax,
                                   lMaxVec=lMaxVec,cleanMask=cleanMask,
                                   sigThresh=sig_thresh[majorIter],verbosity=10,
                                   damp=damp,loopGain=loop_gain,windowSizeDeg=6)
        if not(loopCond):
            # If no sources found then break the loop.
            break