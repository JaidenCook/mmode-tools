__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import warnings
from mmode_tools.plots import plot_baseline_fringes
from scipy.optimize import curve_fit, OptimizeWarning

import numpy as np
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz,EarthLocation
from scipy.optimize import minimize,curve_fit
from astropy.time import Time
from tqdm import tqdm
from astropy import units
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from functools import partial


from mmode_tools.constants import c,MRO,ONSALA

def convert_real_imag_to_complex(gainsRealImag):
    """convert_real_imag_to_complex _summary_

    Parameters
    ----------
    gainsRealImag : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gainsRealImag = np.array(gainsRealImag)
    Ngains = gainsRealImag.size // 2
    amp = gainsRealImag[:Ngains]
    phase = gainsRealImag[Ngains:]
    return amp * np.exp(1j * phase)

def match_filter_vis(covTensor,modelTensor,flagMatrix):
    covTensorVec = covTensor[:,flagMatrix]
    modelTensorVec = modelTensor[:,flagMatrix]
    weight = 1
    numerator = np.sum(weight*covTensorVec*np.conj(modelTensorVec),axis=0)
    denominator = np.sum(weight*modelTensorVec*np.conj(modelTensorVec),axis=0)
    alpha = numerator / denominator
    return alpha

def calc_model_vis(visModRealImag,*gainsRealImag,antPairs=None,refAnt=0,
                   phaseOnly=False,skyTensor=None):
    """calc_model_vis _summary_

    Parameters
    ----------
    visModRealImag : ndarray, shape (2, Nbaseline)
        Model visibilities stacked as [real, imag].
    gainsRealImag : floats
        Gains parameters. If skyTensor is None, contains 2*Ngains values
        (amplitudes and phases). If skyTensor is provided, contains
        2*Ngains-1 + 2*N_sky_terms values.
    antPairs : array-like, optional
        Antenna pairs, by default None
    refAnt : int, optional
        Reference antenna index, by default 0.
    phaseOnly : bool, optional
        If True, fit only phases (amplitudes fixed to 1), by default False.
    skyTensor : ndarray, optional
        Sky tensor with shape (N_sky_terms, 2, Nbaseline) where each element
        is a tensor with the same structure as visModRealImag, by default None.

    Returns
    -------
    ndarray
        Model visibility as flattened real/imag vector.
    """
    if antPairs is None:
        # Current version does not calculate this internally, and must be 
        # provided. This function is only used to fit the Gains, not for 
        # visibility modelling.
        raise ValueError("antPairs must be provided to calc_model_vis.")
    else:
        Ngains = np.unique(antPairs).size

    gainsRealImag = np.array(gainsRealImag).ravel()
    
    # Extract sky terms if skyTensor is provided
    if skyTensor is not None:
        N_sky_terms = skyTensor.shape[0]
        skyTermsAmpPhase = gainsRealImag[-2*N_sky_terms:]
        gainsRealImag = gainsRealImag[:-2*N_sky_terms]
        # Convert amplitude/phase pairs to complex
        skyTermsAmp = skyTermsAmpPhase[:N_sky_terms]
        skyTermsPhase = skyTermsAmpPhase[N_sky_terms:]
        skyTerms = skyTermsAmp * np.exp(1j * skyTermsPhase)
    else:
        skyTerms = None

    # Phase only peel. 
    if phaseOnly:
        ampVec0 = np.ones(Ngains)
        phaseVec0 = gainsRealImag[:Ngains]
    else:
        ampVec0 = gainsRealImag[:Ngains]
        phaseVec0 = gainsRealImag[Ngains:]

    if refAnt >= 0 and refAnt < Ngains:
        indVec = np.delete(np.arange(Ngains),refAnt)

        gainsPhase = np.zeros(Ngains)
        gainsPhase[indVec] = phaseVec0
        gainsPhase[refAnt] = 0.0
        gains = ampVec0 * np.exp(1j * gainsPhase)
    else:
        gains = ampVec0 * np.exp(1j * phaseVec0)

    #
    visMod = visModRealImag[0, :] + 1j * visModRealImag[1, :]

    gainsArr = gains[antPairs[:, 0]] * np.conj(gains[antPairs[:, 1]])
    
    # Calculate model visibility with optional sky terms
    if skyTerms is not None:
        # Convert skyTensor from real/imag format to complex
        skyTensorComplex = skyTensor[:, 0, :] + 1j * skyTensor[:, 1, :]
        skyMod = np.nansum(skyTerms[:, np.newaxis] * skyTensorComplex, axis=0)
        modelVis = visMod * gainsArr + skyMod
    else:
        modelVis = visMod * gainsArr

    return np.stack((modelVis.real, modelVis.imag)).ravel()

def calc_model_vis_jac(visModRealImag, *gainsRealImag, antPairs=None):
    """Analytic Jacobian of calc_model_vis w.r.t. gains parameters [amp..., phase...].

    For baseline k with antennas (p, q):
        f_k = V_k * a_p * a_q * exp(i*(phi_p - phi_q))

    Non-zero derivatives:
        dRe(f_k)/d(a_p)   =  Re(f_k) / a_p
        dRe(f_k)/d(a_q)   =  Re(f_k) / a_q
        dRe(f_k)/d(phi_p) = -Im(f_k)
        dRe(f_k)/d(phi_q) = +Im(f_k)
        dIm(f_k)/d(a_p)   =  Im(f_k) / a_p
        dIm(f_k)/d(a_q)   =  Im(f_k) / a_q
        dIm(f_k)/d(phi_p) = +Re(f_k)
        dIm(f_k)/d(phi_q) = -Re(f_k)

    Parameters
    ----------
    visModRealImag : ndarray, shape (2, Nbaseline)
        Model visibilities stacked as [real, imag].
    gainsRealImag : floats
        Gains parameters: first Nant are amplitudes, next Nant are phases.
    antPairs : ndarray, shape (Nbaseline, 2)
        Antenna index pairs for each baseline.

    Returns
    -------
    J : ndarray, shape (2*Nbaseline, 2*Nant)
        Jacobian matrix, ready for use as jac= in curve_fit.
    """
    gainsRealImag = np.array(gainsRealImag).ravel()
    Nant = gainsRealImag.size // 2
    amp = gainsRealImag[:Nant]
    phase = gainsRealImag[Nant:]
    gains = amp * np.exp(1j * phase)

    visMod = visModRealImag[0, :] + 1j * visModRealImag[1, :]
    gainsArr = gains[antPairs[:, 0]] * np.conj(gains[antPairs[:, 1]])
    modelVis = visMod * gainsArr          # shape (Nbaseline,)

    Nbaseline = modelVis.size
    J = np.zeros((2 * Nbaseline, 2 * Nant))
    ant1 = antPairs[:, 0]
    ant2 = antPairs[:, 1]
    rows = np.arange(Nbaseline)

    mv_re = modelVis.real
    mv_im = modelVis.imag

    # --- Real-output block (rows 0 .. Nbaseline-1) ---
    J[rows, ant1] = mv_re / amp[ant1]   # d Re(f_k) / d a_p
    J[rows, ant2] += mv_re / amp[ant2]   # d Re(f_k) / d a_q
    J[rows, Nant + ant1]  = -mv_im              # d Re(f_k) / d phi_p
    J[rows, Nant + ant2] +=  mv_im              # d Re(f_k) / d phi_q

    # --- Imag-output block (rows Nbaseline .. 2*Nbaseline-1) ---
    J[Nbaseline + rows, ant1] = mv_im / amp[ant1]   # d Im(f_k) / d a_p
    J[Nbaseline + rows, ant2] += mv_im / amp[ant2]   # d Im(f_k) / d a_q
    J[Nbaseline + rows, Nant + ant1] = mv_re              # d Im(f_k) / d phi_p
    J[Nbaseline + rows, Nant + ant2] += -mv_re              # d Im(f_k) / d phi_q

    return J

def _calculate_bounds(Nant, skyTensor=None):
    """Calculate initial parameters and bounds for gain fitting.

    Parameters
    ----------
    Nant : int
        Number of antennas.
    skyTensor : ndarray, optional
        Sky tensor with shape (N_sky_terms, 2, Nbaseline). If provided, the
        parameter vector is extended by ``2 * N_sky_terms`` sky-term
        amplitude/phase parameters.

    Returns
    -------
    p0 : np.ndarray
        Initial parameter guess: [amp_1, ..., amp_Nant, phase_1, ..., phase_(Nant-1), ...]
    boundsLow : np.ndarray
        Lower parameter bounds.
    boundsHigh : np.ndarray
        Upper parameter bounds.
    """
    N_sky_terms = 0 if skyTensor is None else int(skyTensor.shape[0])

    # Parameters: Nant amplitudes + (Nant-1) phases + 2*N_sky_terms sky params
    p0 = np.hstack((
        np.ones(Nant),
        np.zeros(Nant - 1),
        np.ones(N_sky_terms),
        np.zeros(N_sky_terms),
    ))
    boundsLow = np.zeros(2 * Nant - 1 + 2 * N_sky_terms)
    boundsHigh = np.zeros(2 * Nant - 1 + 2 * N_sky_terms)

    # Gain amplitude bounds
    boundsLow[:Nant] = 0
    boundsHigh[:Nant] = np.inf
    #boundsLow[:Nant] = 1 - 1e-6
    #boundsHigh[:Nant] = 1 + 1e-6

    # Gain phase bounds
    boundsLow[Nant:2 * Nant - 1] = -np.pi
    boundsHigh[Nant:2 * Nant - 1] = np.pi

    # Sky-term amplitude/phase bounds (if present)
    if N_sky_terms > 0:
        sky_amp_start = 2 * Nant - 1
        sky_phase_start = sky_amp_start + N_sky_terms
        boundsLow[sky_amp_start:sky_phase_start] = 0
        boundsHigh[sky_amp_start:sky_phase_start] = np.inf
        boundsLow[sky_phase_start:] = -np.pi
        boundsHigh[sky_phase_start:] = np.pi

    return p0, boundsLow, boundsHigh

def _fit_gains_single_time(tInd, visModSlice, visObsSlice, antPairs, p0,
                           maxfev, boundsLow=None, boundsHigh=None, refAnt=7,
                           phaseOnly=False, skyTensor=None):
    """Fit gains for a single time index. Used by calc_DD_gains_parallel."""
    
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    model_fn = partial(calc_model_vis, antPairs=antPairs, refAnt=refAnt,
                       phaseOnly=phaseOnly, skyTensor=skyTensor)
    #jac_fn   = partial(calc_model_vis_jac, antPairs=antPairs)
    Nant = np.unique(antPairs).size
    if refAnt < 0 and refAnt > Nant:
        # In the case where we want to solve gains with a reference.
        raise ValueError(f"refAnt index {refAnt} is out of bounds for {Nant} "
                         f"antennas.")
    else:
        p0, boundsLow, boundsHigh = _calculate_bounds(Nant, skyTensor=skyTensor)
        
        try:
            popt, pcov = curve_fit(
                model_fn,
                visModSlice,
                visObsSlice.ravel(),
                p0=p0,
                bounds=(boundsLow, boundsHigh),
                #jac=jac_fn,
                maxfev=maxfev,
                #nan_policy='omit',
            )
        except (ValueError, RuntimeError) as e:
            print(f"Curve fitting failed for time index {tInd} with error: {e}. "
                  f"Filling with defaults.")
            popt = p0
            N_sky_terms = 0 if skyTensor is None else int(skyTensor.shape[0])
            pcov = np.zeros((2 * Nant - 1 + 2 * N_sky_terms, 
                             2 * Nant - 1 + 2 * N_sky_terms))
    #try:
        
    if refAnt >= 0 and refAnt < Nant:
        # Reassining the gains into a full vector if fit w.r.t a 
        # reference antenna.
        indVec = np.arange(2*Nant)
        indVec = np.delete(indVec, Nant+refAnt)
        poptFull = np.zeros(2*Nant)
        N_sky_terms = 0 if skyTensor is None else int(skyTensor.shape[0])
        if N_sky_terms > 0:
            # The last 2*N_sky_terms are sky terms. For now we delete these,
            # but we may want to keep them in the future.
            skyTerms = popt[-2 * N_sky_terms:-1*N_sky_terms] * np.exp(1j * popt[-N_sky_terms:])
            #print(N_sky_terms)
            #print(skyTerms) 
            popt = popt[:-2 * N_sky_terms]
            pcov = pcov[:-2 * N_sky_terms, :-2 * N_sky_terms]
        poptFull[indVec] = popt
        popt = poptFull
        pcovFull = np.zeros((2*Nant, 2*Nant))
        pcovFull[np.ix_(indVec, indVec)] = pcov
        pcovFull[refAnt, refAnt] = 0.0
        pcovFull[Nant+refAnt, Nant+refAnt] = 0.0
        pcov = pcovFull
    try:
        gains = convert_real_imag_to_complex(popt)
        gains_err = convert_real_imag_to_complex(np.sqrt(np.diag(pcov)))
    except (ValueError,RuntimeError) as e:
        print(f"Curve fitting failed for time index {tInd} with error: {e}. Filling with defaults.")
        print(f"Curve fitting failed for time index {tInd}. Filling with defaults.")
        gains = np.ones(Nant, dtype=np.complex64)
        gains_err = np.zeros(Nant, dtype=np.complex64)
    return gains, gains_err

def _fit_gains_phase_only_single_time(tInd, visModSlice, visObsSlice, antPairs,
                                      maxfev, refAnt=0, verbose=False,
                                      skyTensor=None):
    """Fit phase-only gains for a single time index.

    Fits ``Nant - 1`` phase parameters (amplitudes fixed at unity) relative to
    a reference antenna, then maps the result back to a full complex gains
    vector of length ``Nant``.  Used as a drop-in replacement for
    :func:`_fit_gains_single_time` when only phase calibration is required.

    Parameters
    ----------
    tInd : int
        Time index (used in diagnostic messages only).
    visModSlice : ndarray, shape (2, Nbaseline)
        Model visibilities stacked as ``[real, imag]``.
    visObsSlice : ndarray, shape (2, Nbaseline)
        Observed visibilities stacked as ``[real, imag]``.
    antPairs : ndarray, shape (Nbaseline, 2)
        0-based antenna-index pairs for each baseline, indexing into the
        *unflagged* antenna set.
    maxfev : int
        Maximum number of function evaluations passed to ``curve_fit``.
    refAnt : int, optional
        Index of the reference antenna within the unflagged set (phase fixed
        at zero), by default 0.
    verbose : bool, optional
        If True, print diagnostic messages on fitting failures, by default False.
    fitSkyTerm : bool, optional
        If True, fit a single complex sky term along with phases, by default False.

    Returns
    -------
    gains : ndarray, shape (Nant,), complex64
        Complex gains with unit amplitude and fitted phases.  The reference
        antenna has gain ``1 + 0j``.
    gains_err : ndarray, shape (Nant,), complex64
        Phase uncertainty encoded as ``exp(1j * sigma_phi)`` for each antenna;
        the reference antenna entry is ``1 + 0j`` (zero uncertainty).
    """
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    Nant = np.unique(antPairs).size
    if refAnt < 0 or refAnt >= Nant:
        raise ValueError(
            f"refAnt index {refAnt} is out of bounds for {Nant} antennas."
        )

    # Nant-1 free phases; calc_model_vis(phaseOnly=True) adds back the
    # reference antenna phase (fixed at 0) internally.
    model_fn = partial(calc_model_vis, antPairs=antPairs, refAnt=refAnt,
                       phaseOnly=True, skyTensor=skyTensor)

    if skyTensor is not None:
        Nsky_terms = skyTensor.shape[0]
        boundsLow = np.hstack([np.full(Nant - 1, -np.pi), np.full(Nsky_terms, 0), 
                               np.full(Nsky_terms, -np.pi)])
        boundsHigh = np.hstack([np.full(Nant - 1, np.pi), np.full(Nsky_terms, np.inf), 
                                np.full(Nsky_terms, np.pi)])
        bounds = (boundsLow, boundsHigh)
        p0 = np.hstack([np.zeros(Nant - 1), np.ones(Nsky_terms), np.zeros(Nsky_terms)])
    else:
        p0 = np.zeros(Nant - 1)
        bounds = (np.full(Nant - 1, -np.pi), np.full(Nant - 1, np.pi))

    #try:
    try:
        popt, pcov = curve_fit(
                        model_fn,
                        visModSlice,
                        visObsSlice.ravel(),
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Curve fitting failed for time index {tInd} with error: {e}. "
                f"Filling with defaults.")
        popt = p0
        Nsky_terms = 0 if skyTensor is None else int(skyTensor.shape[0])
        pcov = np.zeros((Nant - 1 + 2 * Nsky_terms, Nant - 1 + 2 * Nsky_terms))
    
    # Reconstruct full-length phase and covariance arrays (size Nant).
    indVec = np.delete(np.arange(Nant), refAnt)
    phasesFull = np.zeros(Nant)
    if skyTensor is not None:
        Nsky_terms = skyTensor.shape[0]
        phasesFull[indVec] = popt[:Nant - 1]
        # Sky terms are in popt[Nant-1:] but we ignore them here.
    else:
        phasesFull[indVec] = popt

    pcovFull = np.zeros((Nant, Nant))
    pcovFull[np.ix_(indVec, indVec)] = pcov[:Nant - 1, :Nant - 1]

    phase_err = np.sqrt(np.diag(pcovFull))
    try:
        gains = np.exp(1j * phasesFull).astype(np.complex64)
        gains_err = np.exp(1j * phase_err).astype(np.complex64)
    except (ValueError, RuntimeError) as e:
        print(f"Phase-only curve fitting failed for time index {tInd} "
                f"with error: {e}. Filling with defaults.")
        gains = np.ones(Nant, dtype=np.complex64)
        gains_err = np.ones(Nant, dtype=np.complex64)

    return gains, gains_err

def calc_DD_gains_parallel(visMod, visObs, returnErrors=False,
                           flagMatrix=None, avg_factor=1,
                           verbose=False, n_jobs=-1,maxfev=1e4,
                           alpha=False, phaseOnly=False, refAnt=7,
                           skyTensor=None):
    from joblib import Parallel, delayed
    # --- averaging, shape setup, flagMatrix, phase rotation (same as before) ---
    if avg_factor > 1:
        def avg_down(arr, factor):
            n = arr.shape[0] // factor
            return arr[:n * factor].reshape((n, factor) + arr.shape[1:]).mean(axis=1)
        visMod = avg_down(visMod, avg_factor)
        visObs  = avg_down(visObs, avg_factor)

    Ntime, Nant = visObs.shape[0], visObs.shape[1]

    if flagMatrix is None:
        flagMatrix = np.ones((Nant, Nant), dtype=bool)
        flagMatrix[visObs[0] == 0] = False   # use first time-slice for shape

    if alpha:
        boolMatrix = visObs[0] != 0
        alpha = match_filter_vis(visObs,visMod,boolMatrix)
        visMod[:,boolMatrix] = visMod[:, boolMatrix]*alpha[None,:]
    
    # 
    visMod = visMod[:, flagMatrix]
    visObs  = visObs[:, flagMatrix]
    antPairs = np.array(np.where(flagMatrix)).T
    antIDs = np.unique(antPairs)

    # Creating an index pair version of the antenna pairs.
    antIndPairs = np.copy(antPairs)
    for i,ant in enumerate(antIDs):
        antIndPairs[antPairs == ant] = i

    visObs  = visObs  * np.exp(-1j * np.angle(visMod))
    visMod  = visMod  * np.exp(-1j * np.angle(visMod))
    if skyTensor is not None:
        skyTensor = skyTensor[:, :, flagMatrix]
        skyTensor = skyTensor * np.exp(-1j * np.angle(visMod))[None,:,:]
    

    # Select fitting function; p0 is only needed for the full (amp+phase) fit.
    if phaseOnly:
        print("Fitting Phase only...")
        fitFunc = _fit_gains_phase_only_single_time
    else:
        fitFunc = _fit_gains_single_time
        p0 = np.hstack((np.ones(antIDs.size), np.zeros(antIDs.size)))

    if verbose:
        verbosityLevel = 10 if n_jobs == -1 else 5
        print(f"Calculating direction-dependent gains for {Nant} antennas and "
              f"{Ntime} time samples using {n_jobs} parallel jobs.")
    else:
        verbosityLevel = 0
    # --- parallel loop over time ---
    if phaseOnly:
        
        results = Parallel(n_jobs=n_jobs, verbose=verbosityLevel)(
            delayed(fitFunc)(
                tInd,
                np.stack((visMod[tInd].real, visMod[tInd].imag)),
                np.stack((visObs[tInd].real, visObs[tInd].imag)),
                antIndPairs, maxfev,
                refAnt=refAnt, skyTensor=skyTensor
            )
            for tInd in range(Ntime)
        )
    else:
        results = Parallel(n_jobs=n_jobs, verbose=verbosityLevel)(
            delayed(fitFunc)(
                tInd,
                np.stack((visMod[tInd].real, visMod[tInd].imag)),
                np.stack((visObs[tInd].real, visObs[tInd].imag)),
                antIndPairs, p0, maxfev,
                refAnt=refAnt, skyTensor=skyTensor
            )
            for tInd in range(Ntime)
        )

    gainsArr = np.ones((Ntime, Nant),dtype=np.complex64)
    gainsErrArr = np.zeros((Ntime, Nant), dtype=np.complex64)
    for tInd, (gains, gains_err) in enumerate(results):
        try:
            gainsArr[tInd,antIDs] = gains.astype(np.complex64)
            gainsErrArr[tInd,antIDs] = gains_err.astype(np.complex64)
        except Exception as e:
            print(f"Error at time index {tInd}: {e}")

    return (gainsArr, gainsErrArr) if returnErrors else gainsArr

def calc_DD_gains(visMod,visObs,returnErrors=False,
                  flagMatrix=None,avg_factor=1,verbose=False,maxfev=100000):
    """calc_DD_gains with optional averaging down of visMod and visObs by avg_factor.

    Parameters
    ----------
    visMod : np.ndarray
        Model visibilities, shape (Ntime, Nant, Nant)
    visObs : np.ndarray
        Observed visibilities, shape (Ntime, Nant, Nant)
    returnErrors : bool, optional
        If True, return gain errors, by default False
    flagAntInds : array-like, optional
        Indices of antennas to flag, by default None
    avg_factor : int, optional
        If >1, average visMod and visObs along time axis by this factor, by default 1

    Returns
    -------
    gainsArr : np.ndarray
        Complex gain solutions, shape (Ntime, Nant)
    gainsErrArr : np.ndarray, optional
        Complex gain errors, shape (Ntime, Nant), if returnErrors is True
    """

    # --- Optionally average down visMod and visObs ---
    if avg_factor > 1:
        def avg_down(arr, factor):
            n = arr.shape[0] // factor
            new_shape = (n, factor) + arr.shape[1:]
            arr = arr[:n*factor]
            return arr.reshape(new_shape).mean(axis=1)
        visMod = avg_down(visMod, avg_factor)
        visObs = avg_down(visObs, avg_factor)

    # Checking whether the data is averaged in time or not.
    Ntime = visObs.shape[0]
    Nant = visObs.shape[1]

    if verbose:
        print(f"Calculating direction-dependent gains for {Nant} antennas and "
              f"{Ntime} time samples.")
        
    # Initialising the gain and gain error arrays.
    gainsArr = np.ones((Ntime, Nant), dtype=np.complex64)
    gainsErrArr = np.zeros((Ntime, Nant), dtype=np.complex64)

    # Option to flag bad known antennas.
    if flagMatrix is None:
        flagMatrix = np.ones((Nant, Nant), dtype=bool)
        flagMatrix[visObs == 0] = False
    
    #
    visMod = visMod[:,flagMatrix]
    visObs = visObs[:,flagMatrix]
    antPairs = np.array(np.where(flagMatrix)).T
    #antIDs = np.unique(antPairs)

    # Phase rotate towards the model source.
    visObs = visObs * np.exp(-1j * np.angle(visMod))
    visMod = visMod * np.exp(-1j * np.angle(visMod))

    initialGains = np.ones(Nant) + 0j * np.ones(Nant)
    for tInd in tqdm(range(Ntime)):
        visModRealImag = np.stack((visMod[tInd].real, visMod[tInd].imag))
        visObsRealImag = np.stack((visObs[tInd].real, visObs[tInd].imag))
        # Performing the curve fitting to solve for the gains.
        try:
            popt, pcov = curve_fit(
                lambda x, y, *g: calc_model_vis(x, y, *g, antPairs=antPairs),
                visModRealImag,
                visObsRealImag.ravel(),
                p0=np.hstack((initialGains.real, initialGains.imag)),
                maxfev=maxfev,
            )    
            # Formatting the fitted gains and their errors into complex form.
            gainsArr[tInd, :] = convert_real_imag_to_complex(popt)
            gainsErrArr[tInd, :] = convert_real_imag_to_complex(np.sqrt(np.diag(pcov)))
        except (ValueError, RuntimeError):
            print(f"Curve fitting failed for time index {tInd}. Filling gains "
                  f"with default values.")
            gainsArr[tInd, :] = 1 + 0j
            gainsErrArr[tInd, :] = 0 + 0j
            pass
    
    if returnErrors:
        return gainsArr, gainsErrArr
    else:
        return gainsArr

def apply_gains(visTensor, gainsArr):
    """apply_gains _summary_

    Parameters
    ----------
    visTensor : _type_
        _description_
    gainsArr : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if visTensor.ndim == 3:
        gainsTensor = np.zeros_like(visTensor)
    elif visTensor.ndim == 4:
        gainsTensor = np.zeros_like(visTensor[0])

    for i, gains in enumerate(gainsArr):
        gainsTensor[i, :, :] = np.outer(gains, np.conj(gains))

    print(gainsTensor.shape, visTensor.shape)
    visCalTensor = np.copy(visTensor)
    if visTensor.ndim == 3:
        visCalTensor = visCalTensor / gainsTensor
    elif visTensor.ndim == 4:
        visCalTensor = visCalTensor / gainsTensor[None, :, :, :]

    return visCalTensor

def calc_resid_tensor(covTensor,modelTensor):

    residualTensor = covTensor*np.exp(-1j*np.angle(modelTensor)) - \
    modelTensor*np.exp(-1j*np.angle(modelTensor))
    residualTensor = residualTensor*np.exp(1j*np.angle(modelTensor))
    residualTensor[np.isnan(residualTensor)] = 0 + 0j

    return residualTensor

def peel(covTensor,t,Array,flags,coords,modelTensor=None,freq=160e6,Ngrid=65,altCut=5,
         verbose=False,returnParams=False,gainsSmooth=False,altCutDD=None,
         window_length=21,polyorder=3,phaseOnly=False,plotGains=False,
         plotFringes=False,DDcal=True,skyTensor=None,dLscale=1,ampSmooth=True,
         maxfev=int(2e4),blineCut=0,gainAmpThresh=10,loopGain=1):
    """peel _summary_   

    Parameters
    ----------
    covTensor : _type_
        _description_
    t : _type_
        _description_
    Array : _type_
        _description_
    flags : _type_
        _description_
    coords : _type_
        _description_
    Ngrid : int, optional
        _description_, by default 65
    altCut : int, optional
        _description_, by default 5
    verbose : bool, optional
        _description_, by default False
    returnParams : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from mmode_tools.caltools import calc_DD_gains_parallel
    from mmode_tools.vistools import calc_DFT_source_vis_model, calc_DFT_source_vis_model_parallel
    from mmode_tools.vistools import alt_cut_func
    from scipy.signal import savgol_filter
    #
    maxBaseline = np.sqrt(Array.uu_m**2 + Array.vv_m**2)[flags].max()
    lam = c/freq
    theta = lam/maxBaseline
    lstVec = t.sidereal_time("apparent").hour
    dL = 2*theta*dLscale
    dM = 2*theta*dLscale

    if verbose:
        print(f"Angular resolution: {theta:.3f} [rads]")
    
    if plotFringes:
        plot_baseline_fringes(lstVec, covTensor, (5,11), interferometer=Array,
                              title='Data Fringes')
    #
    if modelTensor is None:
        blineFlagMatrix = flags * (np.sqrt(Array.uu_m**2 + Array.vv_m**2) > blineCut)
        #params = calc_DFT_source_vis_model(covTensor,t,Array,freq,blineFlagMatrix,coords,
        #                                dL=dL,dM=dM,Ngrid=Ngrid,stdThresh=1,
        #                                plotCond=False,returnAll=True,
        #                                window_size=51,sigma=2,altCut=altCut,
        #                                returnSmooth=ampSmooth)
        params = calc_DFT_source_vis_model_parallel(covTensor,t,Array,freq,
                                                    blineFlagMatrix,coords,
                                                    dL=dL,dM=dM,Ngrid=Ngrid,
                                                    stdThresh=1,returnAll=True,
                                                    window_size=51,sigma=2,
                                                    altCut=altCut,
                                                    returnSmooth=ampSmooth)
        #_,lCentVec,mCentVec,lCentNewVec,mCentNewVec,ampVec,ampVecSmooth,modelTensor \
        #    = params
        lCentVec,mCentVec,lCentNewVec,mCentNewVec,ampVec,ampVecSmooth,modelTensor \
            = params
        outParams = (lCentVec,mCentVec,lCentNewVec,mCentNewVec,ampVec,
                     ampVecSmooth)
    else:
        outParams = (None,None,None,None,None,None,None,None)

    if plotFringes:
        plot_baseline_fringes(lstVec, modelTensor, (5,11), interferometer=Array,
                              title='Model Fringes')

    altaz = coords.transform_to(AltAz(obstime=t,location=MRO))
    altVec = altaz.alt.degree

    # Can either be a tuple/list or int/float. Allows for altitude cut that is 
    # different from horizon to horizon. The beam response is assymetric.
    if isinstance(altCut,(tuple,list)):
        leftAltCut = altCut[0]
        rightAltCut = altCut[1]
    elif isinstance(altCut,(int,float)):
        leftAltCut = altCut
        rightAltCut = altCut

    #
    altBoolVec = alt_cut_func(altVec,leftAltCut,rightAltCut)
    if DDcal:
        blineFlagMatrix = flags * (np.sqrt(Array.uu_m**2 + Array.vv_m**2) > blineCut)
        if altCutDD is not None:
            altBoolVecDD = alt_cut_func(altVec,altCutDD[0],altCutDD[1])
        else:
            altBoolVecDD = altBoolVec
        if skyTensor is not None:
            skyTensor = skyTensor[:, altBoolVecDD, :]
        gainsArr,_ = calc_DD_gains_parallel(modelTensor[altBoolVecDD],
                                            covTensor[altBoolVecDD],
                                            returnErrors=True,flagMatrix=blineFlagMatrix,
                                            avg_factor=1,verbose=verbose,
                                            maxfev=maxfev,phaseOnly=phaseOnly,
                                            skyTensor=skyTensor)
    else:
        # Incase you don't want DD cal. Some timesteps this seems to fail.
        gainsSmooth = False
        plotGains = False
        gainsArr = np.ones((altBoolVec.sum(), covTensor.shape[-1]), 
                           dtype=np.complex64)

    #
    for ind,gainsMat in enumerate(gainsArr):
        boolvec = np.abs(gainsMat) > gainAmpThresh
        gainsArr[ind,boolvec] = 1 + 0j

    # If True smooth the gains for me.
    if gainsSmooth:
        if phaseOnly:
            # Smooth the unwrapped phase to preserve unit amplitude.
            smoothedPhase = savgol_filter(np.unwrap(np.angle(gainsArr), axis=0),
                                          window_length=window_length,
                                          polyorder=polyorder, axis=0,
                                          mode="interp")
            gainsArr = np.exp(1j * smoothedPhase).astype(np.complex64)
        else:
            gainsArr = (savgol_filter(gainsArr.real, window_length=window_length, 
                                      polyorder=polyorder, axis=0, mode="interp")
                                      + 1j * savgol_filter(gainsArr.imag, 
                                                           window_length=window_length, 
                                                           polyorder=polyorder, 
                                                           axis=0, mode="interp")).astype(np.complex64)

    if plotGains:
        antPairs = np.array(np.where(flags)).T
        if altCutDD is not None and DDcal:
            plot_ant_gains(gainsArr,lstVec=lstVec[altBoolVecDD],
                        antPairs=antPairs)
        else:
            plot_ant_gains(gainsArr,lstVec=lstVec[altBoolVec],
                        antPairs=antPairs)
    # Default is 1.
    gainsTensor = np.ones_like(modelTensor)

    #
    if altCutDD is not None and DDcal:
        timeIndVec = np.arange(lstVec.size)[altBoolVecDD]
    else:
        timeIndVec = np.arange(lstVec.size)[altBoolVec]
    for i,tind in enumerate(timeIndVec):
        gainsTensor[tind] = np.outer(gainsArr[i],np.conj(gainsArr[i]))
    
    # Performing the flagging first.
    modelTensorCal = modelTensor*gainsTensor
    modelTensorCal[:,~flags] = 0 + 0j

    #
    residTensor = calc_resid_tensor(covTensor,modelTensorCal*loopGain)
    if returnParams:
        return residTensor, gainsArr, outParams
    else:
        return residTensor, gainsArr


def plot_ant_gains(gainsArr, lstVec=None, antPairs=None):
    """plot_ant_gains Function for plotting the gains solutions for each 
    antenna.

    Parameters
    ----------
    gainsArr : array_like
        Array of complex gain values for each antenna.
    lstVec : array_like, optional
        Array of Local Sidereal Time (LST) values, by default None
    antPairs : array_like, optional
        Array of antenna pairs, by default None
    """

    import matplotlib.pyplot as plt
    if antPairs is not None:
        antIDs = np.unique(antPairs)
    else:
        antIDs = np.arange(gainsArr.shape[1])

    if lstVec is None:
        lstVec = np.arange(gainsArr.shape[0])
    #
    antStrides = np.arange(0,antIDs.size+1,9)
    antIDsSubVec = antIDs[np.random.choice(antIDs.size, size=9, replace=False)]

    for indMin,indMax in zip(antStrides[:-1],antStrides[1:]):
        antIDsSubVec = antIDs[indMin:indMax]
        _, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
        axs = axs.ravel()

        for i, ant in enumerate(antIDsSubVec):
            g = gainsArr[:, ant]
            axs[i].plot(lstVec,np.abs(g), color="k", lw=1.5, label="|g|")
            axs[i].plot(lstVec,g.real, lw=1.2, label="Re(g)")
            axs[i].plot(lstVec,g.imag, lw=1.2, label="Im(g)")
            axs[i].plot(lstVec,np.ones_like(lstVec), lw=1.2, 
                        label="|g|=1", ls="--", color="gray")
            axs[i].set_title(f"Antenna {ant}")
            axs[i].grid(alpha=0.3)

        axs[0].legend(fontsize=9, loc="best")
        for ax in axs[6:]:
            if lstVec is not None:
                ax.set_xlabel("Time Index")
            else:
                ax.set_xlabel("LST [hours]")
        for ax in axs[0::3]:
            ax.set_ylabel("Gain value")

        plt.tight_layout()
        plt.show()