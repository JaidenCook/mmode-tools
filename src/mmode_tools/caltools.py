__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import numpy as np
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz,EarthLocation
from scipy.optimize import minimize,curve_fit
from astropy.time import Time
from tqdm import tqdm
from astropy import units
from scipy.optimize import curve_fit

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
    realPart = gainsRealImag[: len(gainsRealImag) // 2]
    imagPart = gainsRealImag[len(gainsRealImag) // 2 :]
    return realPart + 1j * imagPart

def calc_model_vis(visModRealImag,*gainsRealImag,antPairs=None):
    """calc_model_vis _summary_

    Parameters
    ----------
    visModRealImag : _type_
        _description_
    gainsRealImag : _type_
        _description_
    antPairs : array-like, optional
        Antenna pairs, by default antPairs

    Returns
    -------
    _type_
        _description_
    """
    if antPairs is None:
        # Current version does not calculate this internally, and must be 
        # provided. This function is only used to fit the Gains, not for 
        # visibility modelling.
        raise ValueError("antPairs must be provided to calc_model_vis.")

    gainsRealImag = np.array(gainsRealImag).ravel()
    gains = (
        gainsRealImag[: len(gainsRealImag) // 2]
        + 1j * gainsRealImag[len(gainsRealImag) // 2 :]
    )

    visMod = visModRealImag[0, :] + 1j * visModRealImag[1, :]

    modelVis = np.zeros_like(visMod)
    for ind,(ant1,ant2) in enumerate(antPairs):
        modelVis[ind] = visMod[ind] * gains[ant1] * np.conj(gains[ant2])
    
    return np.stack((modelVis.real, modelVis.imag)).ravel()

def calc_DD_gains(visMod,visObs,returnErrors=False,
                  flagMatrix=None,avg_factor=1,verbose=False):
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
        print(visMod.shape, visObs.shape)
        
    # Initialising the gain and gain error arrays.
    gainsArr = np.ones((Ntime, Nant), dtype=np.complex64)
    gainsErrArr = np.zeros((Ntime, Nant), dtype=np.complex64)

    if verbose:
        print(gainsArr.shape,gainsErrArr.shape)
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
                maxfev=100000,
            )    
            # Formatting the fitted gains and their errors into complex form.
            gainsArr[tInd, :] = convert_real_imag_to_complex(popt)
            gainsErrArr[tInd, :] = convert_real_imag_to_complex(np.sqrt(np.diag(pcov)))
        except ValueError:
            print(f"Curve fitting failed for time index {tInd}. Filling gains with default values.")
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