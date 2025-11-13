import pyshtools
import numpy as np
from tqdm import tqdm
import healpy as hp
import h5py as h5
from warnings import warn
from joblib import Parallel, delayed
from pylops.optimization.basic import cgls
from pylops import MatrixMult,Diagonal

def alm2hpx(alms,nside):
    import healpy as hp

    lmax = 3*nside-1
    maxidx = int(hp.Alm.getidx(lmax,lmax,lmax)) + 1
    print(lmax, maxidx, hp.Alm.getlmax(maxidx))

    alm_array = np.zeros(maxidx, dtype=complex)
    Ncells = int(alms.shape[1])
    for idx in range(maxidx):
        l,m = hp.Alm.getlm(lmax, idx)
        if l < int(Ncells/2):
            alm_array[idx] = alms[0, l, m]

    rot_map = hp.Rotator(rot=[0, 180])
    mymap = rot_map.rotate_map_alms(hp.alm2map(alm_array, nside=nside))
    return mymap


def invert_CGLS_multi_pylops(alm_coeffs_tensor,mmodeTensor,lmax,Niter=10,
                             rtol=1.2,verbosity=0,njobs=-2,damp=0.5,
                             max_nbytes=100e6,weights=None,**kwargs):
    """
    Parallelised function for determining the sky-coefficients.

    Parameters
    ----------
    alm_coeffs_tensor : np.ndarray, np.complex64
        Beam fringe coefficient tensor, should have shape (Nbase,2lmax+1,2lmax+1).
    mmode_tensor : np.ndarray, np.complex64
        Tensor containing the mmode visibilities.
    lmax : int, default=130
        Maximum l-mode, this determines the number of mmodes.
    Niter : int, default=10
        Number of CGLS iterations.
    rtol : float, default=1.2
        Tolerance factor for cgls iteration.
    verbosity : int, defualt=0
        Controls the output of pylops cgls function.
    njobs : int, default=-2
        Determines the number of processes.
    damp : float, default=0.5


    Returns
    -------
    skyCoTensor : np.ndarray, np.complex64
        The recovered sky coefficients.
    """
    # Making sure these values are set to zero, otherwise inversion will fail.
    mmodeTensor[np.isnan(mmodeTensor)] = 0
    mmodeTensor[np.isinf(mmodeTensor)] = 0

    # Allowing for different dampening coefficients for each m-mode.
    if isinstance(damp,float):
        damp = np.ones(lmax+1)*damp
    elif isinstance(damp,np.ndarray):
        if len(damp.shape) > 1:
            raise ValueError("Damp array should have dimensions of 1.")
        if len(damp) != (lmax+1):
            raise ValueError(f"Damp array should have length of {lmax+1}.")

    #skyCoTensor = np.zeros((lmax,lmax),dtype=np.complex64) #slm
    skyCoTensor = np.zeros((lmax+1,lmax+1),dtype=np.complex64) #slm
    def invert_CGLS_permmode(mmode):

        Aop = MatrixMult(alm_coeffs_tensor[:,:,mmode],
                            dtype="complex64")
        if len(mmodeTensor.shape) > 2:
            y = mmodeTensor[:,0,mmode]
        else:
            y = mmodeTensor[:,mmode]
        tol = np.abs(np.std(y))*rtol #damp was 0.1
        xest,_,_,_,_,_ = cgls(Aop,y,x0=np.zeros(lmax+1),damp=damp[mmode], 
                              niter=Niter,tol=tol,show=False)
        skyCoTensor[:,mmode] = np.asarray(np.copy(xest))
                
    _ = Parallel(n_jobs=njobs,require='sharedmem',max_nbytes=max_nbytes,
                 verbose=verbosity)(delayed(invert_CGLS_permmode)(m) \
                                    for m in range(0,lmax+1))
    
    #
    skyCoTensor = restore_negmodes(skyCoTensor[:,:])

    if np.any(np.isinf(skyCoTensor)):
        print('Overflow warning set coefficient values to inf. Setting to 0.')
        skyCoTensor[np.isinf(skyCoTensor)] = 0
    if np.any(np.isnan(skyCoTensor)):
        print('Overflow warning set coefficient values to nan. Setting to 0.')
        skyCoTensor[np.isnan(skyCoTensor)] = 0

    return skyCoTensor

def restore_negmodes(coeffs_positive): #2D matrix of positive m-modes alone
    """
    Uses the conjugate relationship of a real valued sky to recreate the 
    negative m-modes.

    Parameters
    ----------
    coeffs_positive : numpy array, np.complex64
        Numpy array of positive m-mode coefficients.

    Returns
    -------
    coeffs_restored : numpy array, np.complex64
        Numpy array of positive and negative m-modes, positive comes first.
    """
    
    N_lmodes = coeffs_positive.shape[0]
    coeffs_restored = np.zeros([2,N_lmodes,N_lmodes],dtype=np.complex64)
    sign_flipper = (-1)**np.arange(0,N_lmodes)
    coeffs_restored[0,:,:] = coeffs_positive
    coeffs_restored[1,:,:] = np.conj(np.einsum("j,ij->ij",sign_flipper, 
                                               coeffs_positive))
    
    return coeffs_restored

def invert_CGLS_multi_pylops_h5py(filePaths,mmodeTensor,flagMatrixList,
                                  lmax=130,Niter=10,rtol=1.2,verbosity=0,
                                  njobs=-2,damp=0.5,max_nbytes=100e6,
                                  weights=None,**kwargs):
    """
    Performs a parallel calculation of the mmodes for inversion using pylops
    conjugate gradient least squares (cgls). Takes as input the filepaths to 
    the beam fringe maps for each array required to perform the inversion. These
    files are typically large, and large lmax need to be read in one mmode at a
    time. This function is a lazy loaded version of invert_CGLS_multi_pylops.

    Parameters
    ----------
    filePaths : list
        List of file paths for the input beam fringe maps. List can contain one
        or multiple elements.
    mmode_tensor : np.ndarray, np.complex64
        Tensor containing the mmode visibilities.
    flagMatrixList : list
        List containing np.ndarray flag matrices for each array. Matrices must
        be symmetric and square.
    lmax : int, default=130
        Maximum l-mode, this determines the number of mmodes.
    Niter : int, default=10
        Number of CGLS iterations.
    rtol : float, default=1.2
        Tolerance factor for cgls iteration.
    verbosity : int, defualt=0
        Controls the output of pylops cgls function.
    njobs : int, default=-2
        Determines the number of processes.
    damp : float, default=0.5


    Returns
    -------
    skyCoTensor : np.ndarray, np.complex64
        The recovered sky coefficients.
    """
    # Making sure these values are set to zero, otherwise inversion will fail.
    mmodeTensor[np.isnan(mmodeTensor)] = 0
    mmodeTensor[np.isinf(mmodeTensor)] = 0

    # Allowing for different dampening coefficients for each m-mode.
    if isinstance(damp,float):
        damp = np.ones(lmax+1)*damp
    elif isinstance(damp,np.ndarray):
        if len(damp.shape) > 1:
            raise ValueError("Damp array should have dimensions of 1.")
        if len(damp) != (lmax+1):
            raise ValueError(f"Damp array should have length of {lmax+1}.")

    # Initialising the output sky coefficients.
    skyCoTensor = np.zeros([2,lmax+1,lmax+1],dtype=np.complex64) #slm

    # Initialising lists for data sets and baseline IDs.
    dataSetList = []
    blineIDsList = []
    NbaseTot = 0
    for i,filePath in enumerate(filePaths):
        f = h5.File(filePath,'r')
        flagMatrix = flagMatrixList[i]
        Nant = flagMatrix.shape[0]
        # Get the auto correlation flat array indices.
        flatAutoInds = np.ravel_multi_index(np.diag_indices(Nant),
                                            dims=(Nant,Nant))
        # Matrix needs to be flattened.
        flatFlagMatrix = flagMatrix.flatten()
        # Delete the auto-correlations.
        flatFlagMatrix = np.delete(flatFlagMatrix,flatAutoInds)

        # Append the flagged baseline ID list.
        blineIDsList.append(f['data']['blineID'][flatFlagMatrix])
        # Determining the total number of baselines.
        NbaseTot += len(f['data']['blineID'][flatFlagMatrix])

        dset = f['data']['almCoeffTensor']
        dataSetList.append(dset)

    
    def invert_CGLS_permmode(mmode):
        # Initialising total temp beam fringe tensor.
        almTmpTensor = np.zeros((NbaseTot,lmax+1),dtype=np.complex64)

        NbaseSum = 0
        for i,dset in enumerate(dataSetList):
            Nbase = len(blineIDsList[i])
            # Assigning the beam fringes from each array to a total tensor.
            almTmpTensor[NbaseSum:NbaseSum+Nbase,:] = dset[blineIDsList[i],
                                                           :lmax+1,mmode]

            NbaseSum += Nbase

        if np.any(weights):
                # If any weights apply them.
                almTmpTensor = almTmpTensor*weights[:,None]
        # 
        Aop = MatrixMult(almTmpTensor[:,:],
                            dtype="complex64")
        del almTmpTensor
        if len(mmodeTensor.shape) > 2:
            y = mmodeTensor[:,0,mmode]
        else:
            y = mmodeTensor[:,mmode]
        tol = np.abs(np.std(y))*rtol
        xest,_,_,_,_,_ = cgls(Aop,y,x0=np.zeros(lmax+1),damp=damp[mmode],
                                niter=Niter,tol=tol,show=False)
        
        # Assigning the recovered values.
        skyCoTensor[0,:,mmode] = np.asarray(np.copy(xest))
            
    # Performing the inversion.
    _ = Parallel(n_jobs=njobs,require='sharedmem',max_nbytes=max_nbytes,
                 verbose=verbosity)(delayed(invert_CGLS_permmode)(m) \
                                         for m in range(0,lmax+1))
    
    skyCoTensor = restore_negmodes(skyCoTensor[0,:,:])
    if np.any(np.isinf(skyCoTensor)):
        print('Overflow warning set coefficient values to inf. Setting to 0.')
        skyCoTensor[np.isinf(skyCoTensor)] = 0
    if np.any(np.isnan(skyCoTensor)):
        print('Overflow warning set coefficient values to nan. Setting to 0.')
        skyCoTensor[np.isnan(skyCoTensor)] = 0

    return skyCoTensor

def invert_CGLS_multi_pylops_assym(almTensorList,mmodeTensor,
                                   lmax=130,Niter=10,rtol=1.2,verbosity=0,
                                   njobs=-2,damp=0.5,max_nbytes=100e6,
                                   weights=None,**kwargs):
    """
    Performs a parallel calculation of the mmodes for inversion using pylops
    conjugate gradient least squares (cgls). Takes as input the filepaths to 
    the beam fringe maps for each array required to perform the inversion. These
    files are typically large, and large lmax need to be read in one mmode at a
    time. This function is a lazy loaded version of invert_CGLS_multi_pylops.

    Parameters
    ----------
    almTensorList : list
        List of beam fringe almTensors, one for each instrument, or 
        polarisation.
    mmodeTensor : np.ndarray, np.complex64
        Tensor containing the mmode visibilities.
    lmax : int, default=130
        Maximum l-mode, this determines the number of mmodes.
    Niter : int, default=10
        Number of CGLS iterations.
    rtol : float, default=1.2
        Tolerance factor for cgls iteration.
    verbosity : int, defualt=0
        Controls the output of pylops cgls function.
    njobs : int, default=-2
        Determines the number of processes.
    damp : float, default=0.5
    max_nbytes : int, default=100e6
        Defines the amount of available memory for the inversion process.
    weights : float, np.ndarray, float64
        Vector of weights for each baseline. Should be for all input almTensors
        and should have the same size as axis=0 for almTmpMatrix.


    Returns
    -------
    skyCoTensor : np.ndarray, np.complex64
        The recovered sky coefficients.
    """
    # Making sure these values are set to zero, otherwise inversion will fail.
    mmodeTensor[np.isnan(mmodeTensor)] = 0
    mmodeTensor[np.isinf(mmodeTensor)] = 0

    # Shape of the mmode tensor, we only need to solve for the positive m-modes.
    if len(mmodeTensor.shape) > 2:
        mmodeTensor = mmodeTensor[:,0,:]

    # Allowing for different dampening coefficients for each m-mode.
    if isinstance(damp,float):
        damp = np.ones(lmax+1)*damp
    elif isinstance(damp,np.ndarray):
        if len(damp.shape) > 1:
            raise ValueError("Damp array should have dimensions of 1.")
        if len(damp) != (lmax+1):
            raise ValueError(f"Damp array should have length of {lmax+1}.")

    # Initialising the output sky coefficients.
    skyCoTensor = np.zeros([lmax+1,lmax+1],dtype=np.complex64) #slm

    # Initialising lists for data sets and baseline IDs.
    lMaxVec = np.zeros(len(almTensorList),dtype=int)
    NbaseTot = 0
    for i,almTensor in enumerate(almTensorList):
        # Determining the total number of baselines.
        NbaseTot += almTensor.shape[0]
        # lMax can be different for each telescope.
        lMaxVec[i] = int(almTensor.shape[-1])-1

    # Performing check.
    if lmax > np.max(lMaxVec):
         errMsg = f"lmax {lmax} > {np.max(lMaxVec)}, must be strictly smaller"+\
                  f" or equal. Setting lmax to {np.max(lMaxVec)}"
         warn(errMsg)
         #lmax = np.max(lMaxVec)

    def invert_CGLS_permmode(mmode):
        # Initialising total temp beam fringe tensor.
        Bm = np.zeros((NbaseTot,lmax+1),dtype=np.complex64)
        
        NbaseSum = 0
        for i,almTensor in enumerate(almTensorList):
            Nbase = int(almTensor.shape[0])
            # Assigning the beam fringes from each array to a total tensor.
            if mmode <= lMaxVec[i]:
                Bm[NbaseSum:NbaseSum+Nbase,:lMaxVec[i]+1] = \
                    almTensor[:,:lMaxVec[i]+1,mmode]
            else:
                # For assymetric inversion, this is the point where one array
                # no longer has data, so we just assume it is zero.
                pass

            NbaseSum += Nbase

        # 
        Bop = MatrixMult(Bm[:,:],dtype="complex64")
        v = mmodeTensor[:,mmode]
        del Bm

        if np.any(weights):
            Wop = Diagonal(weights,dtype='float64')
            # Creating the weight adjusted operator and data vector.
            Bop = Wop @ Bop
            v = Wop @ v

        tol = np.abs(np.nanstd(v))*rtol
        xest,_,_,_,_,_ = cgls(Bop,v,x0=np.zeros(lmax+1),damp=damp[mmode],
                              niter=Niter,tol=tol,show=False)
        # Assigning the recovered values.
        skyCoTensor[:,mmode] = np.asarray(np.copy(xest))
            
    # Performing the inversion.
    _ = Parallel(n_jobs=njobs,require='sharedmem',max_nbytes=max_nbytes,
                 verbose=verbosity)(delayed(invert_CGLS_permmode)(m) \
                                         for m in range(0,int(lmax+1)))
    
    # Restoring the negartive m-modes.
    skyCoTensor = restore_negmodes(skyCoTensor)

    if np.any(np.isinf(skyCoTensor)):
        print('Overflow warning set coefficient values to inf. Setting to 0.')
        skyCoTensor[np.isinf(skyCoTensor)] = 0
    if np.any(np.isnan(skyCoTensor)):
        print('Overflow warning set coefficient values to nan. Setting to 0.')
        skyCoTensor[np.isnan(skyCoTensor)] = 0
    
    return skyCoTensor

def invert_tikh_multi_assym(almTensorList,mmodeTensor,lmax=130,mmax=None,
                            verbosity=0,
                            njobs=1,damp=0.5,max_nbytes=100e6,
                            weights=None,lMaxVec=None,**kwargs):
    """
    Performs a parallel calculation of the mmodes for inversion using pylops
    conjugate gradient least squares (cgls). Takes as input the filepaths to 
    the beam fringe maps for each array required to perform the inversion. These
    files are typically large, and large lmax need to be read in one mmode at a
    time. This function is a lazy loaded version of invert_CGLS_multi_pylops.

    Parameters
    ----------
    almTensorList : list
        List of beam fringe almTensors, one for each instrument, or 
        polarisation.
    mmodeTensor : np.ndarray, np.complex64
        Tensor containing the mmode visibilities.
    lmax : int, default=130
        Maximum l-mode, this determines the number of mmodes.
    verbosity : int, defualt=0
        Controls the output of pylops cgls function.
    njobs : int, default=-2
        Determines the number of processes.
    damp : float, default=0.5
    max_nbytes : int, default=100e6
        Defines the amount of available memory for the inversion process.
    weights : float, np.ndarray, float64
        Vector of weights for each baseline. Should be for all input almTensors
        and should have the same size as axis=0 for almTmpMatrix.


    Returns
    -------
    skyCoTensor : np.ndarray, np.complex64
        The recovered sky coefficients.
    """
    # Making sure these values are set to zero, otherwise inversion will fail.
    mmodeTensor[np.isnan(mmodeTensor)] = 0
    mmodeTensor[np.isinf(mmodeTensor)] = 0

    # Shape of the mmode tensor, we only need to solve for the positive m-modes.
    if len(mmodeTensor.shape) > 2:
        mmodeTensor = mmodeTensor[:,0,:]

    # Allowing for different dampening coefficients for each m-mode.
    if isinstance(damp,float):
        damp = np.ones(lmax+1)*damp
    elif isinstance(damp,np.ndarray):
        print('Using dampVector.')
        if len(damp.shape) > 1:
            raise ValueError("Damp array should have dimensions of 1.")
        if len(damp) != (lmax+1):
            raise ValueError(f"Damp array should have length of {lmax+1}.")

    # Initialising the output sky coefficients.
    skyCoTensor = np.zeros([lmax+1,lmax+1],dtype=np.complex64) #slm

    # Initialising lists for data sets and baseline IDs.
    if np.any(lMaxVec):
        if len(lMaxVec) != len(almTensorList):
            raise ValueError('len(lMaxVec) != len(almTensorList)')
    else:
        lMaxVec = np.array([int(almTensor.shape[-1])-1 \
                            for almTensor in almTensorList])

    NbaseTot = 0
    for i,almTensor in enumerate(almTensorList):
        # Determining the total number of baselines.
        NbaseTot += almTensor.shape[0]

    # Performing check.
    if lmax > np.max(lMaxVec):
         errMsg = f"lmax {lmax} > {np.max(lMaxVec)}, must be strictly smaller"+\
                  f" or equal. Setting lmax to {np.max(lMaxVec)}"
         warn(errMsg)

    def invert_permmode(m,epsilon=damp):
        # Initialising total temp beam fringe tensor.
        Bm = np.zeros((NbaseTot,(lmax+1)-m),dtype=np.complex64)

        NbaseSum = 0
        for i,almTensor in enumerate(almTensorList):
            Nbase = int(almTensor.shape[0])
            # Assigning the beam fringes from each array to a total tensor.
            if m <= lMaxVec[i]:
                if lMaxVec[i] < lmax:
                    Bm[NbaseSum:NbaseSum+Nbase,:(lMaxVec[i]+1)-m] = \
                        almTensor[:,m:(lMaxVec[i]+1),m]
                else:
                    Bm[NbaseSum:NbaseSum+Nbase,:(lMaxVec[i]+1)] = \
                        almTensor[:,m:(lMaxVec[i]+1),m]
            else:
                # For assymetric inversion, this is the point where one array
                # no longer has data, so we just assume it is zero.
                pass

            NbaseSum += Nbase

        v = mmodeTensor[:,m]
        if np.any(weights):
            if weights.size != Bm.shape[0]:
                raise ValueError(f'Weights shape {weights.size} should match B' +\
                                f' axis 0 size {Bm.shape[0]}')
            else:
                Bm = Bm*weights[:,None]
                v = v*weights

        R = np.diag(epsilon[m]*np.ones(Bm.shape[1]))
        Lam = np.array(np.matrix(Bm).H) @ Bm + R
        #Lam = np.array(np.matrix(Bm).H) @ Bm + epsilon[m:]*np.eye(Bm.shape[1])
        Lam_inv = np.linalg.inv(Lam)
        del Lam

        xest = Lam_inv @ np.array(np.matrix(Bm).H) @ v
        # Assigning the recovered values.
        skyCoTensor[m:,m] = np.asarray(np.copy(xest))
            
    # Performing the inversion.
    if mmax is not None:
        if mmax > lmax:
            raise ValueError(f'mmax {mmax} > lmax {lmax}, must be smaller.')
        else:
            mmax = int(mmax+1)
    else:
        mmax = int(lmax+1)
    _ = Parallel(n_jobs=njobs,require='sharedmem',max_nbytes=max_nbytes,
                 verbose=verbosity)(delayed(invert_permmode)(m) \
                                         for m in range(0,mmax))
    
    # Restoring the negartive m-modes.
    skyCoTensor = restore_negmodes(skyCoTensor)

    if np.any(np.isinf(skyCoTensor)):
        print('Overflow warning set coefficient values to inf. Setting to 0.')
        skyCoTensor[np.isinf(skyCoTensor)] = 0
    if np.any(np.isnan(skyCoTensor)):
        print('Overflow warning set coefficient values to nan. Setting to 0.')
        skyCoTensor[np.isnan(skyCoTensor)] = 0
    
    return skyCoTensor

def filter_coefficients(coeffs,lmax=200,lcut=130,lwin=None,
                        filterType='blackmanharris'):
    """
    Apply Blackman harris filter to input coefficients.

    Parameters
    ----------
    coeffs : np.ndarray, np.complex64
        Array of spherical harmonic coefficients to be filtered.
    lmax : int, default=200
        Maximum spherical harmonic degree to filter to.
    lcut : int
        Spherical harmonic degree of maximum sensitivity.
    lwin : int, default=None
        Window length of taper. Default is lmax-lcut.
    
    Returns
    -------
    None
    """
    ### TODO: Add in more filter types.
    if filterType != 'blackmanharris':
        raise ValueError(f"Filter type {filterType} not implemented, only " +\
                         "blackmanharris is available.")
    else:
        from scipy.signal.windows import blackmanharris
        filterFunc = blackmanharris
    if lcut > lmax:
        errMsg = f"lcut > lmax, should strictly be less."
        raise ValueError(errMsg)

    if (lwin == None) or (lwin > int(lmax-lcut)):
        lwin = int(lmax-lcut)
    
    N = coeffs.shape[1]
    filterBH = np.ones(N)
    filterBH[lcut:lcut+lwin] *= filterFunc(2*lwin)[-lwin:]
    filterBH[lcut+lwin:] = 0
    # Apply the filter to all coefficients.
    coeffs[...] = coeffs*filterBH[None,:,None]

    return None