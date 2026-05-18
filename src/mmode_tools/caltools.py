__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import numpy as np
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz,EarthLocation
from scipy.optimize import minimize
from astropy.time import Time
from tqdm import tqdm
from astropy import units

from mmode_tools.constants import c,MRO,ONSALA

def calc_model_vis(visModRealImag, *gainsRealImag):
    """calc_model_vis _summary_

    Parameters
    ----------
    visModRealImag : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gainsRealImag = np.array(gainsRealImag).ravel()
    gains = (
        gainsRealImag[: len(gainsRealImag) // 2]
        + 1j * gainsRealImag[len(gainsRealImag) // 2 :]
    )

    visMod = visModRealImag[0, :, :] + 1j * visModRealImag[1, :, :]

    modelVis = np.zeros_like(visMod)
    for i in range(visMod.shape[0]):
        for j in range(visMod.shape[1]):
            modelVis[i, j] = visMod[i, j] * gains[i] * np.conj(gains[j])

    return np.stack((modelVis.real, modelVis.imag)).ravel()


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


def calc_gains(visMod, visObs, antIDvec, returnErrors=False, flagAnts=None):
    """calc_gains _summary_

    Parameters
    ----------
    visMod : _type_
        _description_
    visObs : _type_
        _description_
    antIDvec : _type_
        _description_
    returnErrors : bool, optional
        _description_, by default False
    flagAnts : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    # Checking whether the data is averaged in time or not.
    if visObs.ndim == 3:
        Nchan = visObs.shape[0]
    elif visObs.ndim == 4:
        Nchan = visObs.shape[1]

    # Initialising the gain and gain error arrays.
    gainsArr = np.ones((Nchan, len(antIDvec)), dtype=complex)
    gainsErrArr = np.zeros((Nchan, len(antIDvec)), dtype=complex)

    # Option to flag bad known antennas.
    if flagAnts is not None:
        if isinstance(flagAnts, int):
            flagAnts = [flagAnts]
        elif isinstance(flagAnts, np.ndarray) or isinstance(flagAnts, list):
            # Getting the flagged antenna indices.
            flagInds = np.isin(antIDvec, flagAnts)

            # Getting new number of antennas after flagging.
            nside = visMod.shape[-1] - np.sum(flagInds)
            # Reshaping the visibility matrices to remove the flagged antennas.
            visMod = visMod[:, ~flagInds, :][:, :, ~flagInds].reshape(
                (Nchan, nside, nside)
            )
            visObs = visObs[:, ~flagInds, :][:, :, ~flagInds].reshape(
                (Nchan, nside, nside)
            )
    else:
        # If there are no flags then set the flagInds to be all True.
        #flagInds = np.ones(len(antIDvec),dtype=bool)
        flagInds = np.zeros(len(antIDvec),dtype=bool)

    #
    initialGains = np.ones(visMod.shape[1]) + 1j * np.ones(visMod.shape[1])
    for chanInd in range(Nchan):
        visModRealImag = np.stack((visMod[chanInd].real, visMod[chanInd].imag))
        visObsRealImag = np.stack((visObs[chanInd].real, visObs[chanInd].imag))

        # Performing the curve fitting to solve for the gains.
        popt,pcov = curve_fit(calc_model_vis,visModRealImag,
                            visObsRealImag.ravel(),
                            p0=np.hstack((initialGains.real,initialGains.imag)), 
                            maxfev=10000)
        # Formatting the fitted gains and their errors into complex form.
        gainsArr[chanInd, ~flagInds] = convert_real_imag_to_complex(popt)
        gainsErrArr[chanInd, ~flagInds] = convert_real_imag_to_complex(
            np.sqrt(np.diag(pcov))
        )

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