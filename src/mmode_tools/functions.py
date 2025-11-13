import numpy as np

def Gaussian2Dxy(xdata_tuple,amplitude,x0,y0,sigma_x,sigma_y):
    """
    Generalised 2DGaussian function. This 2D Gaussian is fixed to the xy axis,
    rotation is not possible.
    
    Parameters:
    ----------
    xdata_Tuple : tuple
        Tuple containing the X-data and Y-data arrays.
    params : tuple
        Params.
            
    Returns:
    ----------
    g : numpy array
        2D numpy array, the N_Gaussian image.
    """
    (x,y) = xdata_tuple
    x0 = float(x0)
    y0 = float(y0)
    g = amplitude*np.exp(-0.5*(((x-x0)/sigma_x)**2 + ((y-y0)/sigma_y)**2))

    return g