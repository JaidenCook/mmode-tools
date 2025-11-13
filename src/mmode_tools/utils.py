import numpy as np
import h5py as h5
import sys,os
from warnings import warn
from copy import copy,deepcopy
from tqdm import tqdm

from mmode_tools.io import read_LST_VisCube,read_flags,load_beam_fringe_coef
from mmode_tools.io import read_data_config
from mmode_tools.interferometers import RadioArray
from mmode_tools.vistools import vis2mmode_DFT
from mmode_tools.inversion import invert_tikh_multi_assym,filter_coefficients
from mmode_tools.constants import c


def load_alm_tensor_list(beamFilePaths,telescopes,flagMatrixDict,
                         lMax=160,filterParams=None):
    """
    Parameters
    ----------
    beamFilePaths,
    telescopes,
    flagMatrixDict,
    lMax=160,
    filterParams=None

    Returns
    -------
    almTensorList
    """
    almTensorList = []
    for i,beamFilePath in enumerate(beamFilePaths):
        # Appending.
        flagMatrixTemp = flagMatrixDict[telescopes[i]]
        almTemp = load_beam_fringe_coef(beamFilePath,lmax=lMax,
                                        flagMatrix=flagMatrixTemp)
        
        if filterParams is not None and isinstance(filterParams,dict):
            # If given and telescope name is present, then apply a filter to 
            # the blm coefficients.
            try:
                params = filterParams[f"{telescopes[i]}"]
                lcut = params['lcut']
                lwin = params['lwin']
                lmax = params['lmax']
                print(f'Filtering the {telescopes[i]} alm beam fringes.')
                filter_coefficients(almTemp,lmax=lmax,
                                    lcut=lcut,lwin=lwin)
            except KeyError:
                # If telescope is not present do not filter.
                pass
                
        almTensorList.append(almTemp)

    return almTensorList

def load_calc_mmode_tensor(arrFilePaths,Arrays,telescopes,stokesList,
                           flagMatrixDict,lMax=200,freq=160,verbose=False,
                           diffVis=False,returnAntPairs=False):
    """
    Parameters
    ----------
    arrFilePaths,
    Arrays,
    telescopes,
    stokesList,
    flagMatrixDict,
    lMax=200,
    freq=160,
    verbose=False,
    diffVis=False,
    returnAntPairs=False

    Returns
    -------
    mmodeTensor,
    antPairsList
    """
    # Have to use lists because data size is not necessarily the same.
    lam = c/(freq*1e6)
    Ncells = int(2*lMax+2)

    covTensorList = []
    antPairsList = []
    lstVecList = []
    NbaseVec = np.zeros(len(telescopes),dtype=int)
    for i,telescope in enumerate(telescopes):
        flagMatrixTemp = flagMatrixDict[telescopes[i]]
        # Reading in the good antennas, and the flag matrix.
        _,antPairs = RadioArray.get_baselines(Arrays[telescope],calcAutos=False,
                                              flagMatrix=flagMatrixTemp)
        # Appending the flag matrices.
        antPairsList.append(antPairs)
        # Assuming that lstVec is the same for all covariance tensors.
        lstVec,covTensorTemp = read_LST_VisCube(arrFilePaths[i],applyFlags=True,
                                                verbose=verbose,lstSort=True,
                                                flagMatrix=flagMatrixTemp,
                                                stokes=stokesList[i],
                                                diffVis=diffVis,reshape=False)
        
        # Setting the auto correlations to zero.
        Nant = covTensorTemp.shape[-1] 
        covTensorTemp[:,np.arange(Nant),np.arange(Nant)] = 0
        # Appending covTensor to the list. Not the most efficient way to do this,
        # but we have the memory budget.
        covTensorList.append(covTensorTemp)
        lstVecList.append(lstVec)
        NbaseVec[i] = antPairs.shape[0]

    rMatrixList = []
    for telescope in telescopes:
        rMatrixList.append(np.sqrt(Arrays[telescope].uu_m**2+\
                                Arrays[telescope].vv_m**2)/lam)
    # 
    NblineTot = np.sum(NbaseVec)
    #
    mmodeTensor = np.zeros([NblineTot,2,int(Ncells/2)],dtype=np.complex64)
    blineInd = 0
    for i,telescope in enumerate(telescopes):
        # Calculating the m-mode tensor for the current telescope.
        if diffVis:
            # If this is not None, this will set all m > mmax to zero.
            rMatrix = None
        else:
            rMatrix = rMatrixList[i]
        tempMmodeTensor = vis2mmode_DFT(covTensorList[i],lstVecList[i],
                                        np.where(lstVecList[i]>0)[0],Ncells,
                                        rMatrix=rMatrix,plotTest=False)
        # Getting the antenna pairs.
        antPairs = antPairsList[i]
        # Looping through all baseline pairs and assigning to matrix.
        for j in range(antPairs.shape[0]):
            ant1,ant2 = antPairs[j].astype(int)
            try:
                mmodeTensor[blineInd,:,:] += tempMmodeTensor[:,:,ant1,ant2]
            except IndexError:
                errMsg = f"Index error in calculation for {telescope}."
                raise IndexError(f"{errMsg}\n{covTensorList[i].shape}"+\
                                 f"\n{tempMmodeTensor.shape}"+\
                                 f"\n{mmodeTensor[blineInd,:,:].shape}")
            blineInd += 1
    
    if verbose and diffVis:
        print("-------------------------------------")
        print(f'Diff Viss = {diffVis}')
        print(np.mean(mmodeTensor))
        print(np.nanmean(covTensorList[0]))
        print(covTensorList[0].shape)
        print(mmodeTensor.shape)
        print("-------------------------------------")


    if returnAntPairs:
        return mmodeTensor,antPairsList
    else:
        return mmodeTensor

def calc_noise_weights(arrFilePaths,Arrays,telescopes,stokesList,
                       flagMatrixDict,lMax=160,freq=160,verbose=True):
    """
    Calculate the noise weights for the given data.

    Parameters
    ----------
    arrFilePaths,
    Arrays,
    telescopes,
    stokesList,
    flagMatrixDict,
    lMax=160,
    freq=160,
    verbose=True

    Returns
    -------
    """
    from scipy.stats import iqr
    print("Calculating the noise weights.")
    lam = c/(freq*1e6)
    # Diff mmode tensor, required for calculating the noise weights.
    diffMmodeTensor,antPairsList= load_calc_mmode_tensor(arrFilePaths,Arrays,
                                                         telescopes,stokesList,
                                                         flagMatrixDict,
                                                         lMax=lMax,freq=freq,
                                                         verbose=verbose,
                                                         diffVis=True,
                                                         returnAntPairs=True)

    NblineTot = diffMmodeTensor.shape[0]
    blineInd = 0
    noiseWeights = np.zeros(NblineTot)
    testVec = np.zeros(NblineTot)
    for i,telescope in enumerate(telescopes):
        antPairs = antPairsList[i]
        Array = Arrays[telescope]
        for j in range(antPairs.shape[0]):
            ant1,ant2 = antPairs[j].astype(int)
            # Calculating the maximum m-mode index for each baseline.
            # This is the maximum m-mode index that can be calculated for the
            # given baseline.
            mmax = np.ceil(2*np.pi*np.sqrt(Array.uu_m[ant1,ant2]**2 + \
                    Array.vv_m[ant1,ant2]**2)/lam).astype(int)
            noiseWeights[blineInd] = \
                np.nanmedian(np.abs(diffMmodeTensor[blineInd,0,mmax:]))/np.sqrt(2)

            if telescope == 'MWA':
                testVec[blineInd] = 1
            elif telescope == 'EDA2':
                testVec[blineInd] = 2
            
            blineInd += 1
            

    weights = 1/noiseWeights
    # Depening on the lMax some m-modes might be zero.
    weights[np.isnan(weights)] = 0

    del diffMmodeTensor
    return weights


def calc_uniform_weights(arrFilePaths,Arrays,telescopes,verbose=False):
    """
    Calculates uniform weights for the input visibilities.

    Parameters
    ----------
    arrFilePaths,
    Arrays,
    telescopes,
    verbose=False

    Returns
    -------
    weightsVec
    """
    # TODO: Convert this to a Briggs weighting scheme.
    if verbose:
        print('Calculating uniform visibility weights.')

    blineList = []
    blineMaxList = []
    for i,telescope in enumerate(telescopes):
        _,_,flagMatrixTemp,_ = read_flags(arrFilePaths[i])
        flagMatrixTemp[np.diag_indices(flagMatrixTemp.shape[0])] = False
        # Getting the baselines for each of the arrays.  
        blines,_= RadioArray.get_baselines(Arrays[telescope],
                                           flagMatrix=flagMatrixTemp)
        # Calculating the maximum baseline lenght.
        bline_r = np.sqrt(blines[:,0]**2 +blines[:,1]**2)
        # Appending the baselines and the max baseline lengths to associated
        # lists.
        blineList.append(blines)
        blineMaxList.append(np.nanmax(bline_r))
        
    # Creating the grid.
    uMax = int(np.max(np.array(blineMaxList)))+1 # Set absolute grid size to uMax.
    dU = 0.5 # dU = 0.5 grid size for all sky image. Nyquist sample rate.
    Nuv = int(2*uMax/dU) # Number of grid points along one axis. 
    if Nuv % 2 == 0:
        Nuv += 1 # Prefer to be odd numbered.

    # Need to accumulate all the weights first.
    weightArr = np.zeros((Nuv,Nuv),dtype=np.complex64)
    for ind,telescope in enumerate(telescopes):
        uIndVec = np.round((blineList[ind][:,0]+uMax) / dU).astype(int)
        vIndVec = np.round((blineList[ind][:,1]+uMax) / dU).astype(int)

        for i in range(blineList[ind].shape[0]):
            weightArr[vIndVec[i],uIndVec[i]] += 1

    # Calculating the weights vector.
    NbaseVec = np.array([blines.shape[0] for blines in blineList])
    NbaseTot = np.sum(NbaseVec)
    weightsVec = np.zeros(NbaseTot)
    blineInd = 0
    for ind,telescope in enumerate(telescopes):
        # Recalculating the u and v index vectors.
        uIndVec = np.round((blineList[ind][:,0]+uMax) / dU).astype(int)
        vIndVec = np.round((blineList[ind][:,1]+uMax) / dU).astype(int)
        # Looping through each index and getting the associated weight for that
        # visibility.
        for i in range(len(uIndVec)):
            weightsVec[blineInd] = 1/weightArr[vIndVec[i],uIndVec[i]].real
            blineInd += 1

    return weightsVec

def load_data(configFilePath,lMax=160,freq=160,calcWeights=False,
              uniform=False,filterParams=None,verbose=False):
    """

    Parameters
    ----------
    configFilePath,
    lMax=160,
    freq=160,
    calcWeights=False,
    uniform=False,
    filterParams=None,
    verbose=False

    Returns
    -------
    mmodeTensor,
    almTensorList,
    weights
    """
    arrFilePaths,Arrays,telescopes,stokesList,beamFringeFilePaths = \
    read_data_config(configFilePath,returnDates=False)

    flagMatrixDict = {}
    for i,telescope in enumerate(telescopes):
        _,_,flagMatrixTemp,flagBlinesTemp = read_flags(arrFilePaths[i])
        flagMatrixDict[f'{telescope}_flagBlines'] = flagBlinesTemp
        flagMatrixTemp[np.diag_indices(flagMatrixTemp.shape[0])] = False
        try:
            flagMatrixDict[telescope] *= flagMatrixTemp
        except KeyError:
            flagMatrixDict[telescope] = flagMatrixTemp

    mmodeTensor = load_calc_mmode_tensor(arrFilePaths,Arrays,telescopes,
                                         stokesList,flagMatrixDict,lMax=lMax,
                                         freq=freq,verbose=verbose)
    # Loading the alm coefficients.
    almTensorList= load_alm_tensor_list(beamFringeFilePaths,telescopes,
                                        flagMatrixDict,lMax=lMax,
                                        filterParams=filterParams)
    # Calculating the noise weights.
    if calcWeights:
        noiseWeights = calc_noise_weights(arrFilePaths,Arrays,telescopes,stokesList,
                                     flagMatrixDict,lMax=lMax,freq=freq,
                                     verbose=verbose)
        if uniform:
            # If True calculate the uniform weights for the arrays, and combine
            # with the noise weights.
            uniformWeights = calc_uniform_weights(arrFilePaths,Arrays,
                                                  telescopes)
            weights = noiseWeights*uniformWeights
        else:
            weights = noiseWeights

    else:
        weights = None

    return mmodeTensor,almTensorList,weights

def data2map(mmodeTensor,almTensorList,weights,invert=invert_tikh_multi_assym,
             lMax=160,mMax=None,damp=0.01,rtol=1e-4,verbosity=10,njobs=1,
             Niter=10,returnGrid=False,returnCoeffs=False,**kwargs):
    """
    
    Parameters
    ----------
    mmodeTensor,
    almTensorList,
    weights,
    invert=invert_tikh_multi_assym,
    lMax=160,
    mMax=None,
    damp=0.01,
    rtol=1e-4,
    verbosity=10,
    njobs=1,
    Niter=10,
    returnGrid=False,
    returnCoeffs=False,
    **kwargs

    Returns
    -------
    map,
    skyCoTensor,
    raVec,
    decVec
    """
    from pyshtools import SHCoeffs

    if not isinstance(lMax,int):
        lMax = int(lMax)

    lVec = [alm.shape[-1]-1 for alm in almTensorList]

    # Performing the inversion step.
    skyCoTensor=invert(almTensorList,np.conj(mmodeTensor),lmax=lMax,mmax=mMax,
                       rtol=rtol,verbosity=verbosity,damp=damp,njobs=njobs,
                       lMaxVec=lVec,Niter=Niter,weights=weights,**kwargs)
    #
    sphericalCoeffs = SHCoeffs.from_array(skyCoTensor,normalization='ortho',
                                          csphase=-1)
    griddedCoeffs = sphericalCoeffs.expand(grid='DH2',backend='ducc',
                                           lmax_calc=lMax)
    dirtyMap = griddedCoeffs.data

    if returnGrid:
        Nphi = dirtyMap.shape[1]
        dPhi = 360/Nphi
        Ndec = dirtyMap.shape[0]
        dDec = 180/Ndec

        raVec = np.arange(Nphi)*dPhi + dPhi/2
        raVec = np.roll(raVec[::-1],int(raVec.size/2))
        decVec = (np.arange(Ndec) - (Ndec-1)/2)*dDec

        if returnCoeffs:
            return dirtyMap,skyCoTensor,raVec,decVec
        else:
            return dirtyMap,raVec,decVec
    else:
        if returnCoeffs:
            return dirtyMap,skyCoTensor
        else:
            return dirtyMap