__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import numpy as np
from scipy.stats import rayleigh
from scipy.optimize import curve_fit
from numba import njit, prange, get_num_threads, get_thread_id

@njit
def Gaussian2Dxy(xdata_tuple,amplitude,x0,y0,amaj,bmin,theta,
                 normAmp=False):
    """
    Generates 2D Gaussian array.

    Parameters
    ----------
    x : numpy array, float
        2D cartesian or azimuth numpy array. [rad]
    y : numpy array, float
        2D cartesian or zenith numpy array. [rad]
    x0 : numpy array, float
        Cartesian or Azimuth angle of the Gaussian centre. [rad]
    y0 : numpy array, float
        Cartesian or Zenith angle of the centre of the Gaussian. [rad]
    amaj : numpy array, float
        Gaussian major axis. [deg]
    bmin : numpy array, float
        Gaussian minor axis. [deg]
    theta : numpy array, float
        Gaussian position angle. [rad]

    Returns
    -------
    2D Gaussian array.
    """
    (X,Y) = xdata_tuple
    # Defining the width of the Gaussians
    sigx = amaj/(2.0*np.sqrt(2.0*np.log(2.0)))
    sigy = bmin/(2.0*np.sqrt(2.0*np.log(2.0)))

    a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
    b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
    c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)
        
    if normAmp:
        amplitude = amplitude/(2.0*np.pi*sigx*sigy)
    return amplitude*np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))

def power_law(x, amp, x0, index):
    """
    Power law fitting function.
    
    f(x) = amp * (x/x0)^index
    
    Parameters
    ----------
    x : array-like
        Independent variable
    amp : float
        Amplitude normalization factor
    x0 : float
        Reference point for scaling
    index : float
        Power law index (exponent)
    
    Returns
    -------
    array-like
        Power law values
    """
    return amp * (x / x0) ** index


def broken_power_law(x, amp, x0, index1, index2, break_freq):
    """
    Broken power law fitting function with a single break point.
    
    Ensures continuity at the break frequency.
    
    f(x) = amp * (x/x0)^index1                          for x < break_freq
    f(x) = amp * (break_freq/x0)^index1 * (x/break_freq)^index2  for x >= break_freq
    
    Parameters
    ----------
    x : array-like
        Independent variable
    amp : float
        Amplitude normalization at x = x0
    x0 : float
        Reference point for scaling
    index1 : float
        Power law index below the break frequency
    index2 : float
        Power law index above the break frequency
    break_freq : float
        Position of the break point
    
    Returns
    -------
    ndarray
        Broken power law values
    """
    x = np.atleast_1d(x)
    result = np.zeros_like(x, dtype=float)
    
    # Below break frequency
    mask_low = x < break_freq
    result[mask_low] = amp * (x[mask_low] / x0) ** index1
    
    # Above break frequency (continuity enforced)
    mask_high = x >= break_freq
    result[mask_high] = amp * (break_freq / x0) ** index1 * (x[mask_high] / break_freq) ** index2
    
    return result


def offset_rayleigh_pdf(x, scale, x0):
    """Rayleigh distribution offset by x0
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    scale : float
        Scale parameter of the Rayleigh distribution
    x0 : float
        Offset of the distribution
    
    Returns:
    --------
    array-like
        Probability density function values
    """
    return rayleigh.pdf(x - x0, loc=0, scale=scale)

def fit_offset_rayleigh(bins, density, p0=None):
    """
    Fit an offset Rayleigh distribution to histogram data.
    
    Parameters:
    -----------
    bins : array-like
        Bin edges from histogram
    density : array-like
        Density values from histogram
    p0 : tuple, optional
        Initial guess for parameters (scale, x0)
    
    Returns:
    --------
    popt : tuple
        Optimized parameters (scale, x0)
    pcov : ndarray
        Covariance matrix
    """
    # Convert bin edges to bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    
    
    if p0 is None:
        p0 = (1.0, bins[0])
    
    popt, pcov = curve_fit(offset_rayleigh_pdf, bin_centers, density, p0=p0, maxfev=10000)
    
    return popt, pcov