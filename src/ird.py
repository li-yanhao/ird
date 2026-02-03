import numpy as np
import scipy

from .filters import (
    dct_extract_noise, tv_extract_noise, phase_transform, rank_transform
)
from .corr_helper import compute_corr_distances


def preprocess(img:np.ndarray, proc_type:str, proc_param:dict={}) -> np.ndarray:
    if proc_type == None or proc_type.lower() == "none":
        return img

    if proc_type == "phot":
        img = phase_transform(img)
        return img

    if proc_type == "dct":
        if not isinstance(proc_param, dict) or not ("sigma" in proc_param.keys()):
            sigma = 6
        else:
            sigma = proc_param["sigma"]
        img = dct_extract_noise(img, sigma, block_sz=16)
        return img

    if proc_type == "tv":
        img = tv_extract_noise(img, lmda=1)
        return img

    if proc_type == "rt":
        if not isinstance(proc_param, dict) or not ("rt_size" in proc_param.keys()):
            sz = 3
        else:
            sz = proc_param["rt_size"]
        img = rank_transform(img, sz=sz)
        return img

    raise Exception(f" 'proc_type={proc_type}' is not implemented. Check the code for its supported preprocessing type. ")


def count_peaks_with_exclusion(corr_distances_windows:np.ndarray, nb_neighbor:int, excluded_distances:list):
    """Count peaks in correlation windows with exclusion of specified distances.
    
    Parameters
    ----------
    corr_distances_windows : np.ndarray
        Correlation values of shape (nb_distances, nb_windows)
    nb_neighbor : int
        Number of neighboring distances to check when finding peaks
    excluded_distances : list
        List of distances to exclude from peak detection
    
    Returns
    -------
    np.ndarray
        Peak counts for each distance of shape (nb_distances,)
    
    Notes
    -----
    Algorithm steps:
    1. For each window, find local minima along the distance axis
    2. Identify consecutive ranges (bounded by local minima) covering each excluded distance, and nullify them
    3. Validate peak detections on remaining valid correlation distances
    """

    nb_distances, nb_windows = corr_distances_windows.shape
    peak_count = 0

    # valid_mask_distance_window = np.ones((nb_distances, nb_windows), dtype=bool) 
    peak_bool_distance_window = np.zeros((nb_distances, nb_windows), dtype=bool)

    for window in range(nb_windows):
        valid_mask_distance = np.ones(nb_distances, dtype=bool)
        # find local minima which are less than their immediate neighbors
        local_minima = np.where(
                (corr_distances_windows[:, window] < np.roll(corr_distances_windows[:, window], 1)) & \
                (corr_distances_windows[:, window] < np.roll(corr_distances_windows[:, window], -1))
            )[0]

        for i_min in range(1,len(local_minima)-2):
            left, right = local_minima[i_min], local_minima[i_min+1]
            cover_excluded_peak = False
            for excluded_distance in excluded_distances:
                if left <= excluded_distance <= right:
                    cover_excluded_peak = True
                    break
            
            # nullify 3 segments around the excluded peak
            NUM_NULLIFY = 3
            if cover_excluded_peak:
                for j_min in range(i_min-NUM_NULLIFY//2, i_min+NUM_NULLIFY//2+1):
                    left, right = local_minima[j_min], local_minima[j_min+1]
                    valid_mask_distance[left:right+1] = False
        corr_distance_valid = corr_distances_windows[valid_mask_distance, window]
        
        peak_bool_distance_valid = np.zeros_like(corr_distance_valid, dtype=bool)
        for distance_valid in range(nb_neighbor, len(corr_distance_valid)-nb_neighbor):
            if corr_distance_valid[distance_valid] > np.max(corr_distance_valid[distance_valid-nb_neighbor:distance_valid]) and \
                corr_distance_valid[distance_valid] > np.max(corr_distance_valid[distance_valid+1:distance_valid+1+nb_neighbor]):
                peak_bool_distance_valid[distance_valid] = True

        # assign the valid peak bools back to the principle peak bools
        peak_bool_distance_window[valid_mask_distance, window] = peak_bool_distance_valid

    peak_count = np.sum(peak_bool_distance_window, axis=1)
    return peak_count


def compute_nfa_with_exclusion(corr_distance:np.ndarray, nb_neighbor:int, known_peaks:list):
    """ Compute NFA (number of false alarms) for correlation distances with exclusion of known peaks.
    
    Parameters
    ----------
    corr_distance : np.ndarray
        Correlations for all distances and all anchor patches, of shape (nb_distances, nb_windows)
    nb_neighbor : int
        Number of neighboring distances to check on each side when finding peaks
    known_peaks : list
        List of distances to exclude from peak detection
    
    Returns
    -------
    np.ndarray
        NFA values for each distance, of shape (nb_distances,)
    """

    nb_distances = corr_distance.shape[0]
    nb_windows = corr_distance.shape[1]
    peak_counts = count_peaks_with_exclusion(corr_distance, nb_neighbor, known_peaks)
    nfa_binom = np.zeros(nb_distances) + nb_distances
    for distance, count in enumerate(peak_counts):
        p_value = scipy.stats.binom.sf(count, nb_windows, 1/(nb_neighbor*2+1))
        nfa = p_value * nb_distances
        nfa_binom[distance] = nfa

    return nfa_binom



def _detect_resampling_on_residual(img:np.ndarray, window_ratio:float,
                      nb_neighbor:int, direction:str, 
                      is_jpeg:bool, is_demosaic:bool,
                      max_distance:int=-1) -> np.ndarray:
    """ Detect resampling in an image after residual extraction. """
    h, w = img.shape

    if direction == "horizontal":
        nb_distances = w
    else:
        nb_distances = h

    if max_distance == -1: # if not defined
        max_distance = nb_distances // 2 + nb_neighbor + 1

    if direction == "horizontal":
        img = np.array(img.T)

    # compute correlations at distances from 0 to max_distance (inclusive)
    corr_distances = compute_corr_distances(img, window_ratio, max_distance) # (max_distance+1, nb_windows)

    if is_jpeg:
        # excluded_distances should be in the range of [0, max_distance]
        max_order = int(np.floor(max_distance / (nb_distances / 16)))
        excluded_distances = np.arange(1, max_order+1) * (nb_distances / 16)
    elif is_demosaic:
        excluded_distances = np.arange(1,2) * (nb_distances / 2)
    else:
        excluded_distances = []

    nfa_distance = compute_nfa_with_exclusion(corr_distances, nb_neighbor, known_peaks=excluded_distances)

    return nfa_distance


def detect_resampling(img_in:np.ndarray, preproc:str, window_ratio:float,
                      nb_neighbor:int, direction:str, 
                      is_jpeg:bool, is_demosaic:bool,
                      preproc_param:dict=None, max_distance:int=-1, return_preproc_img:bool=False) -> np.ndarray:
    """Detect resampling in image by analyzing distanceic correlations in a given direction.
    
    Parameters
    ----------
    img_in : np.ndarray
        Input grayscale image after residual extraction, of shape (H, W)
    preproc : str
        Type of preprocessing to apply ('none', 'phot', 'dct', 'tv', or 'rt')
    window_ratio : float
        Ratio to determine window size for correlation analysis
    nb_neighbor : int
        Number of neighboring distances to check when finding peaks
    direction : str
        Direction to analyze ('vertical' or 'horizontal')
    is_jpeg : bool
        Whether the image is JPEG compressed, so that distances related to JPEG compression are discarded
    is_demosaic : bool
        Whether the image is demosaiced, so that distances related to demosaicing are discarded
    preproc_param : dict, optional
        Parameters for the preprocessing method
    max_distance : int, optional
        Maximum distance to analyze, defaults to half image size

    Returns
    -------
    np.ndarray
        Array of NFA (number of false alarms) values for each distance
    """

    assert direction in ["vertical", "horizontal"]

    assert len(img_in.shape) == 2

    # step 1: preprocessing
    img = preprocess(img_in, proc_type=preproc, proc_param=preproc_param)
    if return_preproc_img:
        preproc_img = img.copy()
    
    # step 2: compute the correlations of patch pairs
    nfa_distance = _detect_resampling_on_residual(
        img, window_ratio, nb_neighbor, direction,
        is_jpeg, is_demosaic, max_distance
    )

    if return_preproc_img:
        return nfa_distance, preproc_img
    else:
        return nfa_distance


def erode(x:np.ndarray, rg:int):
    x_ero = x.copy()
    for r in range(1, rg+1):
        x_ero[:-r] = np.minimum(x_ero[:-r], x[r:])
        x_ero[r:] = np.minimum(x_ero[r:], x[:-r])

    return x_ero


def cross_validate(nfa_0:np.ndarray, len_0, nfa_1:np.ndarray, len_1, rg:int):
    """ Make agreement between two NFAs, a significant detection from one array must be validated by another significant detection from the other array that is close to the first detection.

    Parameters
    ----------
    nfa_0 : np.ndarray
        the first NFA array
    len_0 : int
        the full length of the first NFA array
    nfa_1 : np.ndarray
        the second NFA array
    len_1 : int
        the full length of the second NFA array
    rg : int
        the range of the cross validation, such that ``out(x) = min_r( max( nfa_0(x), nfa_1(x+r) ) )``
    """

    assert len(nfa_0.shape) == 1 and len(nfa_1.shape) == 1 
    
    if nfa_0.shape != nfa_1.shape:
        # do nearest interpolation on nfa_1 to match nfa_0 shape
        nfa_1 = scipy.interpolate.interp1d(
            np.arange(0, nfa_1.shape[0]) / len_1, nfa_1, kind='nearest', fill_value="extrapolate"
        )(np.arange(0, nfa_0.shape[0]) / len_0)

    # erode nfa_1
    nfa_1_ero = erode(nfa_1, rg)

    return np.maximum(nfa_0, nfa_1_ero)


def detect_resampling_with_cross_val(img_in:np.ndarray, preproc:str, 
                                     window_ratio:float, nb_neighbor:int,
                                     is_jpeg:bool, is_demosaic:bool,
                                     preproc_param:dict=None,
                                     return_preproc_img=False) -> np.ndarray:
    """Detect resampling in image by analyzing distanceic correlations, with cross validation between horizontal and vertical directions.

    Parameters
    ----------
    img_in : np.ndarray
        Input grayscale image as 2D numpy array
    preproc : str
        Type of preprocessing to apply ('none', 'phot', 'dct', 'tv', or 'rt')
    preproc_param : dict, optional
        Parameters for the preprocessing method
    window_ratio : float
        Ratio to determine window size for correlation analysis
    nb_neighbor : int
        Number of neighboring distances to check when finding peaks
    is_jpeg : bool
        Whether the image is JPEG compressed, so that distances related to JPEG compression are discarded
    is_demosaic : bool
        Whether the image is demosaiced, so that distances related to demosaicing are discarded
    return_preproc_img : bool, optional
        Whether to return the preprocessed image

    Returns
    -------
    np.ndarray
        Array of NFA (number of false alarms) values for each distance
    """

    height, width = img_in.shape

    # step 1: preprocessing
    img = preprocess(img_in, proc_type=preproc, proc_param=preproc_param)
    if return_preproc_img:
        preproc_img = img.copy()

    # process horizontal direction
    nfa_h = _detect_resampling_on_residual(
        img, window_ratio, nb_neighbor, direction="horizontal",
        is_jpeg=is_jpeg, is_demosaic=is_demosaic, 
        max_distance=width - 1
    )
    
    # process vertical direction
    nfa_v = _detect_resampling_on_residual(
        img, window_ratio, nb_neighbor, direction="vertical",
        is_jpeg=is_jpeg, is_demosaic=is_demosaic,
        max_distance=height - 1
    )

    # bidirectional cross validation: aspect ratio is kept with small tolerance of 0.5%
    TOL_RATIO = 0.005

    delta_rg = int(TOL_RATIO * width)
    nfa_cv = cross_validate(nfa_h, width, nfa_v, height, delta_rg)

    if return_preproc_img:
        return nfa_cv, preproc_img
    else:
        return nfa_cv
