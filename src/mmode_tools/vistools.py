__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import warnings

import numpy as np
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz,EarthLocation
from scipy.optimize import minimize
from astropy.time import Time
from tqdm import tqdm
from astropy import units
import matplotlib.pyplot as plt

from mmode_tools.constants import c,MRO,ONSALA

def alt_cut_func(altVec,leftCut,rightCut):
    """
    
    """

    maxAltInd = np.argmax(altVec)
    altVecLeft = altVec[:maxAltInd]
    altVecRight = altVec[maxAltInd:]
    boolVecLeft = altVecLeft > leftCut
    boolVecRight = altVecRight > rightCut
    print(boolVecLeft.sum(),boolVecRight.sum())
    altBoolVec = np.concatenate((boolVecLeft,boolVecRight))
    
    return altBoolVec

def vis2mmode_DFT(covTensor,lstVec,goodInds,Ncells,rMatrix=None,IDFT=True,
                  plotTest=False):
    """
    Function to convert visibility tensor to mmodes.

    Parameters
    ----------
    covTensor : numpy array, float
        The visibility tensor
    lstVec : numpy array, float
        Array of all LSTs (including flagged ones)
    goodInds : numpy array, int
        Locations of the good lst data points
    Ncells : int
        Ncells
    IDFT : bool
        Specifies the sign of the Fourier transform in engineering 
        convention. True implies +1, False implies -1.

    Returns
    -------
    mmodeTensor : numpy array, float
        Returns the [2*int(Ncells/2)*Nants*Nants] m-mode tensor
    """
    ## TODO: Change Ncells to lmax int(Ncells/2) = lmax + 1
    L = 24
    lvec = lstVec[goodInds]
    k_plus  = np.arange(0,int(Ncells/2))
    k_minus = -np.arange(0,int(Ncells/2))
    Nants = covTensor.shape[1]
    if IDFT:
        FT_sign = 1
    else:
        FT_sign = -1
    IDFT_matrix = np.zeros([2,int(Ncells/2),len(lvec)], dtype=np.complex64)
    for m in range(int(Ncells/2)):
        IDFT_matrix[0,m,:] = np.exp(FT_sign*2j*np.pi*k_plus[m]*lvec/L)
        IDFT_matrix[1,m,:] = np.exp(FT_sign*2j*np.pi*k_minus[m]*lvec/L)

    mmodeTensor = np.zeros([2,int(Ncells/2),Nants,Nants],dtype=np.complex64)
    mmodeTensor[:,:,:,:] = np.einsum("tac,smt->smac",covTensor[goodInds,:,:],
                                     IDFT_matrix,optimize="optimal")/len(lvec)
    
    # Setting all visibility values greater than the associated lMax to zero.
    if rMatrix is not None:
        # Assume that rMatrix has the same shape as mmodeTensor.
        # Assume that the units of rMatrix are in wavelengths.
        lMaxMatrix = (2*np.pi*rMatrix).astype(int) + 1
        kPlusTensor = np.ones_like(mmodeTensor[0,:,:,:],
                                   dtype=int)*k_plus[:,None,None]
        kMinusTensor = np.ones_like(mmodeTensor[0,:,:,:],
                                    dtype=int)*np.abs(k_minus[:,None,None])
        
        if kPlusTensor.shape != (int(Ncells/2),Nants,Nants):
            print(kPlusTensor.shape)
            print(mmodeTensor.shape[1:])
            print((int(Ncells/2),Nants,Nants))
            raise ValueError("kPlusTensor shape does not match mmodeTensor shape.")

  
        boolTensor = kPlusTensor >= lMaxMatrix[None,:,:]
        mmodeTensor[0,boolTensor] = 0 + 0j
        boolTensor = kMinusTensor > lMaxMatrix[None,:,:]
        mmodeTensor[1,boolTensor] = 0 + 0j
  
        if plotTest:
            # Plotting test thresholded m-mode visibilities
            from mmode_tools.flag import plot_autos
            from matplotlib import pyplot as plt
            antInd1,antInd2 = 28,11
            antInd1,antInd2 = 5,11
            testVec = mmodeTensor[1,:,antInd1,antInd2]
            testVecConj = mmodeTensor[1,:,antInd2,antInd1]

            fig,axs = plt.subplots(1,figsize=(10,5))
            figaxs = (fig,axs)

            plot_autos(k_plus,np.abs(testVec),figaxs=figaxs,color='k')
            plot_autos(k_plus,np.abs(testVecConj),figaxs=figaxs,color='tab:red')
            axs.plot([lMaxMatrix[antInd1,antInd2],lMaxMatrix[antInd1,antInd2]],
                    [0,np.abs(testVec).max()],color='k',ls='--',zorder=1e2,lw=3)
            axs.set_title(f'm-mode Visibilities: r = {rMatrix[antInd1,antInd2]:5.3f}' + \
                        r' [$\lambda$],' + f' (ant1,ant2) = {antInd1,antInd2}')

            axs.set_xlabel('m',fontsize=18)
            axs.set_yscale('linear')

            axs.grid()
            axs.legend(labels=[r'$|\mathcal{V}_m|$',r'$|\mathcal{V}^*_m|$',
                            r'$l_\mathrm{max}$'],
                    ncols=2)
            axs.set_yscale('log')

    return mmodeTensor

def DFT_image(covMatrix,Naxis,antLoc,freq):
    """
    Function to generate post-correlation beamformed images. i.e. DFT images.

    Parameters
    ----------
    covMatrix : numpy array, complex
        The Hermitian correlation matrix
    Naxis : int
       required size of the map, assumes a square map 
    antLoc : numpy array, float
        antenna array locations
    freq : float
        system frequency in Hz

    Returns
    -------
    skyImg : numpy array, float
        The generated map
    """

    lVec = np.linspace(-1.0,1.0,Naxis)
    mVec = np.linspace(-1.0,1.0,Naxis)
    lm = np.zeros([2,Naxis,Naxis],dtype=np.float64)
    lam = c/freq

    for i in range(Naxis):
        for j in range(Naxis):
            l = lVec[i]
            m = mVec[j]
            if (l**2 + m**2 < 1.0):
                lm[:,i,j] = [l,m]

    beamVech = np.exp(-1j*2*np.pi*np.einsum('ij,jkl->ikl',antLoc[:,0:2],lm)/lam)
    
    skyImg = np.einsum("ijk,il,ljk->jk",beamVech,covMatrix,
                       np.conj(beamVech),optimize='optimal')
    return skyImg

def max_cov_gains(lmOff,SunAlt,SunAz,dataTensor,antLocs,lam):
    """
    Calculates the squared sum of the covariance gains towards a given direction
    used to find the maximum offset in direction cosine space.
    
    Parameters
    ----------
    lmOff : tuple
        Contains the x and y offsets in direction cosine space.
    SunAlt : float
        Sun altitude in radians.
    SunAz : float
        Sun azimuth in radians.
    dataTensor : np.complex64, np.ndarray
        Covariance matrix.
    antLocs : float, np.ndarray
        List of antenna locations in east and north.
    lam : float
        Observing wavelength.
    
    Returns
    -------
    covGainSum : float
        squared sum of the covariance gains towards the sun.
    """
    l0,m0 = lmOff
    lSun = np.cos(SunAlt)*np.sin(SunAz) - l0 #- np.random.normal(0,0.01,1)[0]
    mSun = np.cos(SunAlt)*np.cos(SunAz) - m0 #- np.random.normal(0,0.01,1)[0]
    lm_Sun = np.array([lSun,mSun])
    wVec = np.exp(-1j*2*np.pi*np.einsum('ij,j->i',antLocs,lm_Sun)/lam)
    
    measCovGains = np.einsum('i,il,l->i',wVec,dataTensor,
                             np.conj(wVec),optimize='optimal')
    
    #return 1/np.abs(np.sum(measCovGains**2))
    return 1/np.sum(np.abs(measCovGains))

def master_holocal_Sun(covTensor,tgpsVec,timeInd,location,radioArray,
                       freq,SunFluxI=31913,blineMin=4,blineMax=1e6,
                       beamVal=0.5,refAntInd=2,
                       verbose=False):
    """
    Using the sun perform holography to get the gain amplitude and phase 
    solutions for each antenna.

    Parameters
    ----------
    covTensor : np.complex64, np.ndarray
        Visibility covariance tensor, first axis is the time axis, second and 
        third axis are the associated covariance matrix.
    tgpsVec : float, np.ndarray
        Vector containing the UTC times for each covariance matrix in GPS 
        format. 
    timeInd : int, np.ndarray
        Calibration time index.
    location : astropy location object
        Astropy location object, default input should be the MRO.
    radioArray : Radio_array object
        mmode_tools array object, default should be the EDA2.
    freq : float
        Observation frequency in Hz.
    SunFluxI : float, default=31913
        Sun flux density.
    blineMin : float, default=4
        Minimum baseline length in m. 
    blineMax : float, default=1e6
        Maximum baseline length in m.
    verbose : bool, default=False
        Output parameter, if True print output information.
    fitlmoff : bool, default=False
        If True fit the offset position of the source in l and m.
    delta : float, default=0.05
        lm-offset value, max is 1.
    
    Returns
    -------
    complexGains
    """
    lam = c/freq
    covMatrix = np.copy(covTensor[timeInd,:,:])
    Nants = covMatrix.shape[0]

    # Getting the antenna locations.
    antLoc = np.column_stack([radioArray.east,radioArray.north])
    # Getting the good antenna pairs after baseline flagging.
    _,goodAntPairs = radioArray.get_baselines(radioArray,blineMin=blineMin,
                                              blineMax=blineMax)
    #In this matrix, 1 indicates a good antenna pair, while 0 is for a bad 
    # antenna pair
    flagMat = np.zeros([Nants,Nants]) 
    for antPair in goodAntPairs:
        ant0 = int(antPair[0])
        ant1 = int(antPair[1])
        flagMat[ant0,ant1] = 1.0
        flagMat[ant1,ant0] = 1.0

    time = Time(tgpsVec[timeInd],format="gps",scale='ut1')
    altazframe = AltAz(obstime=time,location=location)

    # Getting the solar altitude and azimuth.
    srcAltAz = get_sun(time).transform_to(altazframe)
    srcAlt = np.radians(srcAltAz.alt.degree)
    srcAz = np.radians(srcAltAz.az.degree)

    # Fitting the offset in the sun alt and az positions.
    lmMin = minimize(max_cov_gains,(0,0),
                     args=(srcAlt,srcAz,covMatrix*flagMat,antLoc,lam),
                     method="Nelder-Mead",bounds=((-0.1,0.1),(-0.1,0.1)))
    lOff,mOff = lmMin.x
    if verbose:
        print(f"lm offset = {lOff:5.3f},{mOff:5.3f}")
        print(f'Sun altitude = {np.degrees(srcAlt):5.3f} [deg]')
        
    # Source direction cosine vector.
    lmSrc = np.array([np.cos(srcAlt)*np.sin(srcAz)-lOff,
                      np.cos(srcAlt)*np.cos(srcAz)-mOff])
    # Weights vector.
    wVec = np.exp(-1j*2*np.pi*np.einsum('ij,j->i',antLoc,lmSrc)/lam)

    # Covariance matrix gains.
    covGains = np.einsum('i,il,il,l->i',wVec,covMatrix,flagMat,np.conj(wVec),
                         optimize='optimal')
    covGains = covGains/covGains[refAntInd]
    # Covariance gains phases.
    GainsPhase = np.angle(covGains)
    phaseCalMatrix = np.exp(-1j*GainsPhase) # Phase rotation matrix.

    # Absolute Gain amplitudes.
    absGains = np.abs(covGains)
    ampCalMatrix = np.divide(np.ones(Nants),absGains,out=np.zeros(Nants), 
                             where=absGains>0.5)

    # Calculating the calibrated covariance matrices.
    covMatPhaseCal = np.einsum('a,ab,b->ab',phaseCalMatrix,covMatrix,
                             np.conj(phaseCalMatrix),optimize='optimal')
    covMatAmpCal = np.einsum('a,ab,b->ab',ampCalMatrix,covMatPhaseCal,
                             ampCalMatrix,optimize='optimal')
    covMatAmpCalPhased = np.einsum('i,il,l-> il',wVec,covMatAmpCal,
                                   np.conj(wVec),optimize='optimal')
    SunFluxPreCal = np.real(np.nansum(covMatAmpCalPhased*flagMat)/np.nansum(flagMat))
    
    fluxFac = (SunFluxI*beamVal)/SunFluxPreCal
    if fluxFac < 0:
        fluxFac = np.abs(fluxFac)

    ampCalMatrix = ampCalMatrix*np.sqrt(fluxFac)
    covMatPhaseCal = np.einsum('a,ab,b->ab',phaseCalMatrix,covMatrix,
                               np.conj(phaseCalMatrix),optimize='optimal')

    covMatAmpCal = np.einsum('a,ab,b->ab',ampCalMatrix,covMatPhaseCal,
                             ampCalMatrix,optimize='optimal')
    covMatAmpCalPhased = np.einsum('i,il,l-> il',wVec,covMatAmpCal,
                                   np.conj(wVec),optimize='optimal')
    SunFluxPostCal = np.real(np.nansum(covMatAmpCalPhased*flagMat)/np.nansum(flagMat))

    if verbose:
        print(f"Sun flux pre calibration is {SunFluxPreCal:5.3f} Jy")
        print(f"Sun flux post calibration is {SunFluxPostCal:5.3f} Jy")
        print(f"Beam value = {beamVal}")

    complexGains = ampCalMatrix*phaseCalMatrix
    return complexGains

def apply_cal_sols(covTensor,gainSols):
    """
    Apply the calibration solutions to the covariance tensor.
    
    Parameters
    ----------
    covTensor : np.complex64, np.ndarray
        Visibility covariance tensor, first axis is the time axis, second and 
        third axis are the associated covariance matrix.
    gainSols : np.complex64, np.ndarray

    Returns
    -------
    covTensorCal : np.complex64, np.ndarray
    """
    ampCalMatrix = np.abs(gainSols)
    phaseCalMatrix = np.exp(-1j*np.angle(1/gainSols))

    covTensorCal = np.einsum('a,iab,b->iab',phaseCalMatrix,covTensor,
                             np.conj(phaseCalMatrix),optimize='optimal')
    covTensorCal = np.einsum('a,iab,b->iab',ampCalMatrix,covTensorCal,
                             ampCalMatrix,optimize='optimal')
    
    covTensorCal[np.isnan(covTensorCal)] = 0

    return covTensorCal

def calc_mean_covTensor(lstVec,covTensor,Nlst=1440,binsCond=False,
                        tgpsVec=None,Array=None,freq=160,location=MRO):
    """
    Down sample the covariance tensor to a coarser LST grid.

    Parameters
    ----------
    lstVec : numpy array, float
        1D numpy array containing the LST values.
    covTensor : np.complex64, np.ndarray
        Visibility covariance tensor, first axis is the time axis, second and 
        third axis are the associated covariance matrix.
    Nlst : int, default=1440
        Number of grid points to average to, 1440 is the number of minues
        in a day (not a sidereal day).
    tgpsVec : np.ndarray, default=None
        UTC time vector (in gps time format) for each LST bin. Required to 
        calculate the zenith phase rotation tensor.
    Array : array_layouts.Radio_array object
        Mmode tools array object, contains information on baselines and array
        location. Used for modelling, required to perform the phase correction.
    freq : float, default=160
        The frequency of the observation in MHz, required to peform the phase
        correction.
    location : astropy.Earthlocation object, default=MRO
        Required to determine the alt and az for each phase centre, needed for
        the phase correction.
     
    Returns
    -------
    lstAvgVec : numpy array, float
        1D numpy array containing the new LST values.
    covAvgTensor : np.complex64, np.ndarray
        The time average covariance tensor.
    tgpsAvgVec : numpy array, float64 (optional)
        The averaged UTC times for each average LST bin, in GPS time format.
    binVec : numpy array, int (optional)
        Vector containing the number of data points averaged for each bin. Only
        returned if binsCond=True.
    """
    from tqdm import tqdm
    L = 24
    dLST = L/Nlst

    lstGridLow = np.arange(Nlst)*dLST - dLST/2
    lstGridHi = np.arange(Nlst)*dLST + dLST/2

    lstAvgVec = np.zeros(Nlst)
    binVec = np.zeros(Nlst)

    if Array:
        print("Calculating zenith phase correction tensor...")
        # If Array is given assume phase correction is being applied.
        from mmode_tools.modelling import phase_rot_tensor,radec2lmn
        if np.any(tgpsVec):
            # If tgpsVec is not None, then we need to average the tgps values
            # as well.
            tgpsAvgVec = np.zeros(Nlst)
        else:
            raise ValueError('Argument tgpsVec required for phase correction.')

        lam = c/(freq*1e6)
        #L = 23.9344696 # Number of hours in a sidereal day.
        raPhaseVec = np.degrees(2*np.pi*lstVec/L) # RA of each phase time step.
        decPhaseVec = MRO.lat.value*np.ones(lstVec.size)

        # Calculating the direction cosines for each time steps zenith phase 
        # centre.
        lVec,mVec,nVec = radec2lmn(tgpsVec,raPhaseVec,decPhaseVec)
        # Calculating the phase tensor for each baseline and timestep.
        phaseTensor = phase_rot_tensor(Array,lam,lVec,mVec,nVec)

        # Applying phase correction.
        covTensor = covTensor*phaseTensor
        print("Phase correction tensor appliied.")

    # Create the average covariance Tensor object.
    covTensorAvg = np.zeros((Nlst,) + covTensor.shape[1:],dtype=np.complex64)

    for i in tqdm(range(Nlst)):
        lstBool = (lstVec > lstGridLow[i])&(lstVec <= lstGridHi[i])
        lstAvgVec[i] = 0.5*(lstGridLow[i] + lstGridHi[i])
        binVec[i] = lstBool[lstBool].size
        covTensorAvg[i,:,:] = np.nanmean(covTensor[lstBool,:,:],axis=0)

    if np.any(tgpsVec):
        # If tgpsVec is not None, then we need to average the tgps values
        # as well.
        from scipy.interpolate import interp1d
        tgpsInterp = interp1d(lstVec,tgpsVec,fill_value='extrapolate')
        tgpsAvgVec = tgpsInterp(lstAvgVec)

    if np.any(np.isnan(covTensorAvg)):
        covTensorAvg[np.isnan(covTensorAvg)] = 0 + 0j

    if binsCond:
        if np.any(tgpsVec):
            # If tgpsVec is not None, then we need to average the tgps values
            # as well.
            return lstAvgVec,covTensorAvg,tgpsAvgVec,binVec
        else:
            # If True return the number of data points averaged for each grid 
            # point.
            return lstAvgVec,covTensorAvg,binVec
    else:
        if np.any(tgpsVec):
            # If tgpsVec is not None, then we need to average the tgps values
            # as well.
            return lstAvgVec,covTensorAvg,tgpsAvgVec
        else:
            # If tgpsVec is None, then we don't need to average the tgps values
            # as well.
            return lstAvgVec,covTensorAvg

def calc_std_covTensor(lstVec,covTensor,Nlst=1440,binsCond=False,
                       tgpsVec=None,Array=None,freq=160,location=MRO):
    """
    Down sample the covariance tensor to a coarser LST grid.

    Parameters
    ----------
    lstVec : numpy array, float
        1D numpy array containing the LST values.
    covTensor : np.complex64, np.ndarray
        Visibility covariance tensor, first axis is the time axis, second and 
        third axis are the associated covariance matrix.
    Nlst : int, default=1440
        Number of grid points to average to, 1440 is the number of minues
        in a day (not a sidereal day).
    
    Returnsa
    -------
    lstAvgVec : numpy array, float
        1D numpy array containing the new LST values.
    covTensorStd : np.complex64, np.ndarray
        The time average covariance tensor.
    binVec : numpy array, int (optional)
        Vector containing the number of data points averaged for each bin. Only
        returned if binsCond=True.
    """
    from tqdm import tqdm
    L = 24
    dLST = L/Nlst

    indVec = np.arange(covTensor.shape[0])
    lstGridLow = np.arange(Nlst)*dLST - dLST/2
    lstGridHi = np.arange(Nlst)*dLST + dLST/2

    if Array:
        print("Calculating zenith phase correction tensor...")
        # If Array is given assume phase correction is being applied.
        from mmode_tools.modelling import phase_rot_tensor,radec2lmn
        if not(np.any(tgpsVec)):
            # If tgpsVec is not None, then we need to average the tgps values
            # as well.
            raise ValueError('Argument tgpsVec required for phase correction.')

        lam = c/(freq*1e6)
        L = 24 # Number of hours in a sidereal day.
        raPhaseVec = np.degrees(2*np.pi*lstVec/L) # RA of each phase time step.
        decPhaseVec = MRO.lat.value*np.ones(lstVec.size)

        # Calculating the direction cosines for each time steps zenith phase 
        # centre.
        lVec,mVec,nVec = radec2lmn(tgpsVec,raPhaseVec,decPhaseVec)
        # Calculating the phase tensor for each baseline and timestep.
        phaseTensor = phase_rot_tensor(Array,lam,lVec,mVec,nVec)

        # Applying phase correction.
        covTensor = covTensor*phaseTensor
        print("Phase correction tensor appliied.")

    lstAvgVec = np.zeros(Nlst)
    binVec = np.zeros(Nlst)
    # Create the average covariance Tensor object.
    diffCovTensorCalAvg = np.zeros((Nlst,)+covTensor.shape[1:],dtype=np.complex64)
    for i in tqdm(range(Nlst)):
        lstBool = (lstVec > lstGridLow[i])&(lstVec <= lstGridHi[i])
        indVecTemp = indVec[lstBool]        
        N = lstBool[lstBool].size
        
        if N % 2 == 0:
            # If even add an extra point to the end of the vector.
            indVecTemp = np.concatenate((indVecTemp,
                                         np.array([indVecTemp[-1]+1])))
        
        tempEvenCovTensor = covTensor[indVecTemp[::2],:,:]
        tempOddCovTensor = covTensor[indVecTemp[::2]+1,:,:]
        diffCovTensorCalAvg[i,:,:] = np.nanmean(tempEvenCovTensor-tempOddCovTensor,
                                                axis=0)

        lstAvgVec[i] = np.nanmean(lstVec[lstBool])
        
        binVec[i] = indVecTemp.size

    if np.any(np.isnan(diffCovTensorCalAvg)):
        diffCovTensorCalAvg[np.isnan(diffCovTensorCalAvg)] = 0 + 0j

    if binsCond:
        # If True return the number of data points averaged for each grid point.
        return lstAvgVec,diffCovTensorCalAvg,binVec
    else:
        return lstAvgVec,diffCovTensorCalAvg

from scipy.optimize import least_squares
def fit_point_source_enu_offsets(vis, u, v, w, A0, E0, N0,
                                 dE_max, dN_max, weights=None):
    """
    Fit complex amplitude and small ENU position offsets about an image-derived guess.

    Model:
        V = A * exp[-2pi i (uE + vN + wU)]
    with
        E = E0 + dE
        N = N0 + dN

    Parameters
    ----------
    vis : complex ndarray
    u, v, w : ndarray
        Baseline coords in wavelengths (ENU basis)
    A0 : complex or float
        Initial amplitude guess
    E0, N0 : float
        Initial ENU source position from image peak
    dE_max, dN_max : float
        Max absolute offset allowed in E and N
    weights : ndarray, optional
    """

    vis = np.asarray(vis)
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)

    if weights is None:
        weights = np.ones_like(u, dtype=float)
    else:
        weights = np.asarray(weights)

    sqrtw = np.sqrt(weights)
    A0 = complex(A0)

    def model(params):
        Ar, Ai, dE, dN = params
        E = E0 + dE
        N = N0 + dN
        r2 = E*E + N*N
        if r2 >= 1.0:
            U = 1e-12
        else:
            U = np.sqrt(1.0 - r2)

        A = Ar + 1j*Ai
        phase = -2j * np.pi * (u*E + v*N + w*U)
        return A * np.exp(phase)

    def residual(params):
        mod = model(params)
        r = sqrtw * (vis - mod)
        return np.concatenate([r.real, r.imag])

    x0 = np.array([A0.real, A0.imag, 0.0, 0.0], dtype=float)

    lb = [-np.inf, -np.inf, -dE_max, -dN_max]
    ub = [ np.inf,  np.inf,  dE_max,  dN_max]

    result = least_squares(residual, x0, bounds=(lb, ub), method="trf")

    Ar, Ai, dE, dN = result.x
    A_fit = Ar + 1j*Ai
    E_fit = E0 + dE
    N_fit = N0 + dN

    return result, A_fit, E_fit, N_fit, dE, dN

def calc_vis_DFT(uvwArr,lVec,visVec,grid,dL=0.1,dM=0.1,Ngrid=31,stdThresh=2.5,
                 plotCond=False,returnlCent=False,phaseCond=False,
                 returnVisModel=True):
    """calc_vis_DFT _summary_

    Parameters
    ----------
    uvwArr : (N, 3) ndarray
        Baseline coordinates in wavelengths, in ENU basis.
    lVec : (2,) ndarray
        Initial EN direction cosine guess from the image peak.
    visVec : (N,) complex ndarray
        Observed visibilities for one time/frequency chunk.
    grid : (2, Ngrid, Ngrid) ndarray
        Grid coordinates for the DFT calculation.
    dL : float, optional
        Grid spacing in the l direction, by default 0.1
    dM : float, optional
        Grid spacing in the m direction, by default 0.1
    Ngrid : int, optional
        Number of grid points in each direction, by default 31
    stdThresh : float, optional
        Standard deviation threshold for masking, by default 2.5
    plotCond : bool, optional
        Whether to plot the DFT results, by default False
    returnlCent : bool, optional
        Whether to return the l center coordinate, by default False

    Returns
    -------
    visVecModel : (N,) complex ndarray
        Modeled visibilities for the source.
    lCent, mCent : floats
        Center coordinates of the source in the l and m directions.
    beamImg : (Ngrid, Ngrid) ndarray
        Beam image of the source.
    """
    from scipy.stats import iqr
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    uVec = uvwArr[:,0]
    vVec = uvwArr[:,1]
    wVec = uvwArr[:,2]

    l0 = lVec[0]
    m0 = lVec[1]

    dl = dL/Ngrid
    dm = dM/Ngrid

    xGrid,yGrid = grid
    lGrid = l0 + dl*xGrid
    mGrid = m0 + dm*yGrid
    nGrid = np.sqrt(1 - lGrid**2 - mGrid**2)

    uu_lmod = uVec[:,None]*lGrid[None,:]
    vv_mmod = vVec[:,None]*mGrid[None,:]
    if phaseCond:
        ww_nmod = wVec[:,None]*(nGrid[None,:]-1) # Correct way.
    else:
        ww_nmod = wVec[:,None]*nGrid[None,:]

    phaseTensor = np.exp(2*np.pi*1j*(uu_lmod+vv_mmod+ww_nmod))

    visArrPhased = visVec[:,None]*phaseTensor
    IntVec = np.nanmean(visArrPhased,axis=0).real

    intStd = iqr(IntVec[np.isnan(IntVec)==False])

    intBoolInds = (IntVec > stdThresh*intStd) * (np.isnan(nGrid) == False)
    N = IntVec[intBoolInds].size

    if returnlCent:
        lCent = lGrid[np.nanargmax(IntVec)]
        mCent = mGrid[np.nanargmax(IntVec)]

    if plotCond:
        IntVec[intBoolInds==False] = np.nan
        beamImg = IntVec.reshape(Ngrid,Ngrid)
        im = plt.imshow(beamImg.T,
                        origin='lower',
                        extent=[lGrid.min(),lGrid.max(),mGrid.min(),mGrid.max()])
        plt.scatter(l0,m0)
        if returnlCent:
            plt.scatter(lCent,mCent,color='r',marker='x')
        plt.colorbar(im)
        plt.show()

    uu_lmod = uu_lmod[:,intBoolInds]
    vv_mmod = vv_mmod[:,intBoolInds]
    ww_nmod = ww_nmod[:,intBoolInds]

    srcPhaseTensor = np.exp(-2*np.pi*1j*(uu_lmod+vv_mmod+ww_nmod))

    IntVec[intBoolInds==False] = np.nan
    IntMatrix = IntVec[None,intBoolInds]*srcPhaseTensor
    beamImg = IntVec.reshape(Ngrid,Ngrid)
    visVecModel = np.sum(IntMatrix,axis=1)/N

    if returnlCent:
        if returnVisModel:
            return visVecModel,lCent,mCent,beamImg
        else:
            return lCent,mCent,beamImg
    else:
        if returnVisModel:
            return visVecModel,beamImg
        else:
            return beamImg

def calc_DFT_source_vis_model(dataTensor,t,interferometer,freq,flagMatrix,
                              coords,dL=0.025,dM=0.025,Ngrid=65,stdThresh=1,
                              plotCond=False,returnAll=False,phaseCond=False,
                              window_size=51,sigma=5,altCut=0.05,
                              returnSmooth=False):
    """calc_DFT_source_vis_model _summary_

    Parameters
    ----------
    dataTensor : _type_
        _description_
    t : _type_
        _description_
    interferometer : _type_
        _description_
    freq : _type_
        _description_
    flagMatrix : _type_
        _description_
    coords : _type_
        _description_
    dL : float, optional
        _description_, by default 0.025
    dM : float, optional
        _description_, by default 0.025
    Ngrid : int, optional
        _description_, by default 65
    stdThresh : int, optional
        _description_, by default 1
    plotCond : bool, optional
        _description_, by default False
    returnAll : bool, optional
        _description_, by default False
    window_size : int, optional
        _description_, by default 51
    sigma : int, optional
        _description_, by default 5
    altCut : float, optional
        _description_, by default 0.05

    Returns
    -------
    _type_
        _description_
    """
    from astropy.coordinates import AltAz
    from mmode_tools.functions import Gaussian2Dxy
    from mmode_tools.modelling import calc_lmn
    from mmode_tools.utils import gaussian_smooth_1d
    from matplotlib import patches

    lam = c/(freq)
    # Get thge alt for each time step, we will use this to filter out low
    #  altitude timesteps with poor signal.
    altaz = coords.transform_to(AltAz(obstime=t,location=MRO))
    altVec = altaz.alt.degree
    if isinstance(altCut,tuple):
        leftCut,rightCut=altCut
        altBoolVec = alt_cut_func(altVec,leftCut,rightCut)
    elif isinstance(altCut,float) or isinstance(altCut,int):
        altBoolVec = altVec > altCut
    else:
        raise ValueError("altCut must be a float, int or tuple.")
    # Calc uvw's for the post corr beamforming.
    uvwArr = np.vstack((interferometer.uu_m[flagMatrix]/lam,
                        interferometer.vv_m[flagMatrix]/lam,
                        interferometer.ww_m[flagMatrix]/lam)).T
    # Make a grid to perform the DFT on, centered on the phase centre.
    grid = np.mgrid[-Ngrid//2:Ngrid//2,-Ngrid//2:Ngrid//2] + 1
    #grid = np.mgrid[-Ngrid//2:Ngrid//2,-Ngrid//2:Ngrid//2]
    gridFlat = (np.copy(grid)[0].T.flatten(), np.copy(grid)[1].T.flatten())

    # Initialise all the output vectors.
    indVec = np.arange(t.gps.size)
    lCentVec = np.zeros(t.gps.size)
    lCentNewVec = np.zeros(t.gps.size)
    mCentNewVec = np.zeros(t.gps.size)
    mCentVec = np.zeros(t.gps.size)
    poptArr = np.zeros((indVec[altBoolVec].size,6))
    #ampVec = np.zeros(t.gps.size)
    ampVec = np.zeros(t.gps.size,dtype=np.complex64)

    lCentVec,mCentVec,_ = calc_lmn(altaz.alt.value,altaz.az.value,degrees=True)
    # Setting some parameters for the fitting.
    dl = dL/Ngrid
    dm = dM/Ngrid
    sigMax = Ngrid//6 + 1
    # Loop through each time step, and make a DFT grid.
    for i,ind in enumerate(tqdm(indVec[altBoolVec])):
        visVec = dataTensor[ind,flagMatrix]
        lCent,mCent = lCentVec[ind],mCentVec[ind]
        
        #
        _,beamImg = calc_vis_DFT(uvwArr,(lCent,mCent),visVec,gridFlat,dL=dL,
                                dM=dM,Ngrid=Ngrid,stdThresh=stdThresh,
                                plotCond=False,returnlCent=False)
        
        # Getting the important values position and peak intensity.
        try:
            lGrid = lCent + dl*grid[0]
            mGrid = mCent + dm*grid[1]
            # The x Grid might need to be flipped.
            lCentNewVec[ind] = lGrid.T.flatten()[np.nanargmax(beamImg)]
            mCentNewVec[ind] = mGrid.T.flatten()[np.nanargmax(beamImg)]
            peak = np.nanmax(beamImg)

            _, peakNew, lCentNew, mCentNew, dE, dN = fit_point_source_enu_offsets(vis=visVec,
                                                            u=uvwArr[:,0],
                                                            v=uvwArr[:,1],
                                                            w=uvwArr[:,2],
                                                            A0=peak,
                                                            E0=lCentNewVec[ind],
                                                            N0=mCentNewVec[ind],
                                                            dE_max=0.5*dL/Ngrid,
                                                            dN_max=0.5*dM/Ngrid
                                                        )

            # Try a second iteration to see if that leads to any improvement.
            _, peakNew, lCentNew, mCentNew, dE, dN = fit_point_source_enu_offsets(vis=visVec,
                                                            u=uvwArr[:,0],
                                                            v=uvwArr[:,1],
                                                            w=uvwArr[:,2],
                                                            A0=peakNew,
                                                            E0=lCentNew,
                                                            N0=mCentNew,
                                                            dE_max=0.5*dL/Ngrid,
                                                            dN_max=0.5*dM/Ngrid
                                                        )
            lCentNewVec[ind] = lCentNew
            mCentNewVec[ind] = mCentNew
            peak = peakNew

        except (ValueError, UnboundLocalError):
            lCentNewVec[ind] = lCent
            mCentNewVec[ind] = mCent
            peak = 0
        
        # We want to make sure points that lie outside of this radius are zero.
        if np.sqrt((lCent-lCentNewVec[ind])**2 + \
                   (mCent-mCentNewVec[ind])**2) > 0.95*max(dL,dM)/2:
            lCentNewVec[ind] = lCent
            mCentNewVec[ind] = mCent
            peak = 0

        # Assigning parameter values.
        popt = np.array([peak,0,0,sigMax,sigMax,1.127])
        poptArr[i] = popt
        #lCentNewVec[ind] = lCent + popt[1]*dL/Ngrid
        #mCentNewVec[ind] = mCent + popt[2]*dM/Ngrid
        
        lCentVec[ind] = lCent
        mCentVec[ind] = mCent
        ampVec[ind] = peak
        
        if plotCond:
            if i%100 == 0:
                plot = True 
            else:
                plot = False
        else:
            plot = False

        if plot:
            # Creating the l and m  grid.    
            print(f"Ind {ind}: alt {altaz.alt.value[ind]}, az {altaz.az.value[ind]}, l {lCent}, m {mCent}")
            #print(f"dl {lCentNewVec[ind]-lCent}, dm {mCentNewVec[ind]-mCent}",np.sqrt((0.5*dL)**2 + (0.5*dM)**2),dL/2)
            print(f"Fitted params: amp {popt[0]}, x0 {popt[1]}, y0 {popt[2]}, sigx {popt[3]}, sigy {popt[4]}, PA {popt[5]}")
            modImg = Gaussian2Dxy((grid[0].T.flatten(),
                                   grid[1].T.flatten()),
                                   *popt).reshape(beamImg.shape)

            fig,axs = plt.subplots(1,3,figsize=(18,5))
            im1 = axs[0].imshow(beamImg,
                            origin='lower',
                            extent=[lGrid.min(),lGrid.max(),mGrid.min(),mGrid.max()])
            im2 = axs[1].imshow(modImg,
                            origin='lower',
                            extent=[lGrid.min(),lGrid.max(),mGrid.min(),mGrid.max()])
            im3 = axs[2].imshow(beamImg-modImg,
                            origin='lower',
                            extent=[lGrid.min(),lGrid.max(),mGrid.min(),mGrid.max()])
            axs[0].scatter(lCent,mCent)
            # Peak brightness position.
            axs[0].scatter(lCentNewVec[ind],mCentNewVec[ind],color='r',marker='x')
            axs[1].scatter(lCentNewVec[ind],mCentNewVec[ind],color='r',marker='x')
            fig.colorbar(im1, ax=axs)
            ellipse = patches.Ellipse(
                xy=(lCentNewVec[ind],mCentNewVec[ind]),
                width=popt[3]*dL/Ngrid,
                height=popt[4]*dM/Ngrid,
                angle=360-np.degrees(popt[5]),# matplotlib uses degrees
                edgecolor='red',
                facecolor='none',
                linewidth=1.5,
                label=f'Source Fit'
            )
            axs[0].add_patch(ellipse)
            
            plt.show()
    #
    ampVecSmooth = gaussian_smooth_1d(ampVec,window_size=window_size,
                                      sigma=sigma)
    if returnSmooth:
    
        modelVisTensor = make_DFT_model_vis_tensor(interferometer,lam,
                                                   lCentNewVec,mCentNewVec,
                                                   ampVecSmooth,
                                                   phaseCond=phaseCond)
    else:
        modelVisTensor = make_DFT_model_vis_tensor(interferometer,lam,
                                                   lCentNewVec,mCentNewVec,
                                                   ampVec,
                                                   phaseCond=phaseCond)
    #
    if returnAll:
        return (poptArr,lCentVec,mCentVec,lCentNewVec,mCentNewVec,ampVec,
                ampVecSmooth,modelVisTensor)
    else:
        return poptArr,modelVisTensor


def calc_DFT_source_vis_model_parallel(dataTensor,t,interferometer,freq,flagMatrix,
                                       coords,dL=0.025,dM=0.025,Ngrid=65,
                                       stdThresh=1,returnAll=False,
                                       phaseCond=False,window_size=51,sigma=5,
                                       altCut=0.05,returnSmooth=False,
                                       fitCond=True):
    from astropy.coordinates import AltAz
    from mmode_tools.modelling import calc_lmn
    from mmode_tools.utils import gaussian_smooth_1d
    from joblib import Parallel, delayed
    
    


    lam = c/(freq)
    # Get thge alt for each time step, we will use this to filter out low
    #  altitude timesteps with poor signal.
    altaz = coords.transform_to(AltAz(obstime=t,location=MRO))
    altVec = altaz.alt.degree
    if isinstance(altCut,tuple):
        leftCut,rightCut=altCut
        altBoolVec = alt_cut_func(altVec,leftCut,rightCut)
    elif isinstance(altCut,float) or isinstance(altCut,int):
        altBoolVec = altVec > altCut
    else:
        raise ValueError("altCut must be a float, int or tuple.")
    # Calc uvw's for the post corr beamforming.
    uvwArr = np.vstack((interferometer.uu_m[flagMatrix]/lam,
                        interferometer.vv_m[flagMatrix]/lam,
                        interferometer.ww_m[flagMatrix]/lam)).T
    # Make a grid to perform the DFT on, centered on the phase centre.
    grid = np.mgrid[-Ngrid//2:Ngrid//2,-Ngrid//2:Ngrid//2] + 1
    gridFlat = (np.copy(grid)[0].T.flatten(), np.copy(grid)[1].T.flatten())

    # Initialise all the output vectors.
    indVec = np.arange(t.gps.size)
    lCentVec = np.zeros(t.gps.size)
    lCentNewVec = np.zeros(t.gps.size)
    mCentNewVec = np.zeros(t.gps.size)
    mCentVec = np.zeros(t.gps.size)
    ampVec = np.zeros(t.gps.size,dtype=np.complex64)

    lCentVec,mCentVec,_ = calc_lmn(altaz.alt.value,altaz.az.value,degrees=True)
    # Setting some parameters for the fitting.
    dl = dL/Ngrid
    dm = dM/Ngrid
    # Loop through each time step, and make a DFT grid.
    results = Parallel(n_jobs=-1,verbose=1)(delayed(calc_vis_DFT)(uvwArr,
                                                        (lCentVec[ind],
                                                         mCentVec[ind]),
                                                         dataTensor[ind,flagMatrix],
                                                         gridFlat,dL=dL,dM=dM,
                                                         Ngrid=Ngrid,
                                                         stdThresh=stdThresh,
                                                         plotCond=False,
                                                         returnlCent=False,
                                                         returnVisModel=False,)
                                   for ind in indVec[altBoolVec])
    
    for i, ind in enumerate(tqdm(indVec[altBoolVec])):
        beamImg = results[i]
        
        # Getting the important values position and peak intensity.
        try:
            lGrid = lCentVec[ind] + dl*grid[0]
            mGrid = mCentVec[ind] + dm*grid[1]
            # The x Grid might need to be flipped.
            #lCentNewVec[ind] = lGrid.T.flatten()[np.nanargmax(beamImg)]
            #mCentNewVec[ind] = mGrid.T.flatten()[np.nanargmax(beamImg)]
            #ampVec[ind] = np.nanmax(beamImg)
            lCentNewVec[ind] = lGrid.T.flatten()[np.nanargmax(np.abs(beamImg))]
            mCentNewVec[ind] = mGrid.T.flatten()[np.nanargmax(np.abs(beamImg))]
            ampVec[ind] = np.nanmax(np.abs(beamImg))

            if fitCond:
                visVec = dataTensor[ind,flagMatrix]
                Niter = 3
                for _ in range(Niter):
                    _, peakNew, lCentNew, mCentNew, dE, dN = fit_point_source_enu_offsets(vis=visVec,
                                                                    u=uvwArr[:,0],
                                                                    v=uvwArr[:,1],
                                                                    w=uvwArr[:,2],
                                                                    A0=ampVec[ind],
                                                                    E0=lCentNewVec[ind],
                                                                N0=mCentNewVec[ind],
                                                                dE_max=0.5*dL/Ngrid,
                                                                dN_max=0.5*dM/Ngrid
                                                            )
                    if np.sqrt(lCentNew**2 + mCentNew**2) > 1:
                        # If the value is not realistic, then we will just
                        # continue, and use the old values.
                        continue
                    else:
                        lCentNewVec[ind] = lCentNew
                        mCentNewVec[ind] = mCentNew
                        ampVec[ind] = peakNew
        except (ValueError, UnboundLocalError):
            lCentNewVec[ind] = lCentVec[ind]
            mCentNewVec[ind] = mCentVec[ind]
            ampVec[ind] = 0
    
    ampVecSmooth = gaussian_smooth_1d(ampVec,window_size=window_size,
                                      sigma=sigma)
    
    if returnSmooth:
    
        modelVisTensor = make_DFT_model_vis_tensor(interferometer,lam,
                                                   lCentNewVec,mCentNewVec,
                                                   ampVecSmooth,
                                                   phaseCond=phaseCond)
    else:
        modelVisTensor = make_DFT_model_vis_tensor(interferometer,lam,
                                                   lCentNewVec,mCentNewVec,
                                                   ampVec,
                                                   phaseCond=phaseCond)
    #
    if returnAll:
        return (lCentVec,mCentVec,lCentNewVec,mCentNewVec,ampVec,
                ampVecSmooth,modelVisTensor)
    else:
        return modelVisTensor

def make_DFT_model_vis_tensor(interferometer,lam,lCentNewVec,mCentNewVec,
                              ampVec,phaseCond=False):
    """make_DFT_model_vis_tensor _summary_

    Parameters
    ----------
    interferometer : _type_
        _description_
    lam : _type_
        _description_
    lCentNewVec : _type_
        _description_
    mCentNewVec : _type_
        _description_
    ampVec : _type_
        _description_
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    visTensor : numpy.ndarray
        Numpy array containing the modeled visibilities.
    """
    from mmode_tools.modelling import point_mod
    nCentNewVec = np.sqrt(1-lCentNewVec**2 - mCentNewVec**2)

    visTensor = point_mod(interferometer,lam,lCentNewVec,mCentNewVec,
                          nCentNewVec,ampVec,phaseCond=phaseCond)
    
    return visTensor