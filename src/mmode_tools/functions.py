import numpy as np

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