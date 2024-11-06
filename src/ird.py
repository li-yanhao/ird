import numpy as np
import scipy

import filters
from corr_helper import compute_corr_periods



def _bool_lateral_maxima(data, axis=0, order=1, exclude_range=0, lateral='both', mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    exclude_range : int, optional
        When set to n, the nearest n elements are excluded for the comparison. Default is 0.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See numpy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin

    Examples
    --------
    >>> testdata = np.array([1,2,3,4,2,1,0])
    >>> _bool_lateral_maxima(testdata, axis=0, order=1, exclude_range=1, lateral="both")
    array([False, False,  True, True, False, False, False], dtype=bool)

    """

    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1+exclude_range, order+1+exclude_range):
        if lateral == 'left':
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= np.greater(main, minus)
        elif lateral == 'right':
            plus = data.take(locs + shift, axis=axis, mode=mode)
            results &= np.greater(main, plus)
        elif lateral == 'both':
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= np.greater(main, plus)
            results &= np.greater(main, minus)
        if(~results.any()):
            return results
    return results


def preprocess(img:np.ndarray, proc_type:str, **args) -> np.ndarray:
    if proc_type == None or proc_type.lower() == "none":
        return img

    if proc_type == "2nd":
        dir = args["dir"]
        if dir == "horizontal":
            img = filters.derivative(img, order=2)
        elif dir == "vertical":
            img = filters.derivative(img.T, order=2).T
        else:
            raise Exception(f"Variable `dir` must be 'vertical' or 'horizontal', got '{dir}' instead.")

    if proc_type == "dct":
        assert 'sigma' in args.keys(), f"DCT denoising filter must be companied with parameter 'sigma=...' "
        sigma = args["sigma"]
        img = filters.dct_extract_noise(img, sigma, block_sz=16)
        return img

    if proc_type == "dct_8x8":    
        img = filters.dct_extract_noise_8x8(img, 6)

    if proc_type == "tv":
        img = filters.tv_extract_noise(img, lmda=1)

    if proc_type == "dncnn":
        dncnn = filters.DnCNN()
        img = dncnn.extract_noise(img)
        h, w = img.shape[0:2]
        img = img[:h - h%8, :w - w%8]

    if proc_type == "rt":
        img = filters.rank_transform(img, sz=1)
        return img

    if proc_type == "rt_v2":
        img = filters.rank_transform_v2(img, sz=1)
        return img

    if proc_type == "rt_2x2":
        img = filters.rank_transform_nxn(img, n=2)
        return img

    if proc_type == "rt_4x4":
        img = filters.rank_transform_nxn(img, n=4)
        return img

    if proc_type == "rt_8x8":
        img = filters.rank_transform_align_8x8(img, remove_dc=False)
        return img

    if proc_type == "rt_no_dc":
        img = filters.rank_transform_no_dc(img, sz=3)
        return img

    if proc_type == "krt":
        img = filters.kernel_rank_transform(img, sz=1, sigma=0.3)
        return img

    if proc_type == "rgt":
        img = filters.rank_gaussian_transform(img, sz=1)
        return img

    if proc_type == "rt_accum":
        img = filters.rank_transform_accum(img, sz=3)
        return img

    if proc_type == "srt":
        img = filters.squeezed_rank_transform(img, sz=1)
        return img
    
    raise Exception(f" 'proc_type={proc_type}' is not implemented. Check the code for its supported preprocessing type. ")


def detect_resampling(img_in:np.ndarray, preproc:str, window_ratio:float, nb_neighbor:int, direction:str, is_jpeg:bool, max_period_tested:int=-1):
    """_summary_

    Parameters
    ----------
    img_in : np.ndarray
        one channel image, in shape (h, w)
    preproc : str
        _description_
    window_ratio : float
        _description_
    nb_neighbor : int
        _description_
    direction : str
        _description_
    is_jpeg : bool
        _description_
    max_period_tested : int, optional
        _description_, by default -1

    Returns
    -------
    _type_
        _description_
    """

    assert direction in ["vertical", "horizontal"]

    assert len(img_in.shape) == 2

    # step 1: preprocessing
    img = preprocess(img, proc_type=preproc)

    # step 2: compute the correlations of patch pairs

    h, w = img_in.shape

    if direction == "vertical":
        nb_periods = h
    else:
        nb_periods = w

    if max_period == -1: # if not defined
        max_period = nb_periods // 2 + nb_neighbor + 1

    corr_periods = []
    if direction == "horizontal":
        img = np.array(img_in.T)
    else:
        img = img_in

    corr_periods = compute_corr_periods(img, window_ratio, max_period) # period in [0, max_period]

    # step 3: find out the abnormal correlations at certain periods
    nb_windows = corr_periods.shape[1]

    mask_valid_period = np.ones(max_period_tested, dtype=bool)
    mask_valid_period[0] = False
    if is_jpeg:
        for i in range(1, 8):
            period = int(np.round(i / 8 * nb_periods))

            # assume the size is multiple of 8
            if period < max_period_tested:
                mask_valid_period[period-2] = False
                mask_valid_period[period-1] = False
                mask_valid_period[period] = False
                mask_valid_period[period+1] = False
                mask_valid_period[period+2] = False

    corr_periods_valid = corr_periods[mask_valid_period, :]

    peak_mask = _bool_lateral_maxima(corr_periods_valid, axis=0, order=nb_neighbor, exclude_range=1, lateral='both')
    peak_counts_valid = np.sum(peak_mask, axis=1)

    # set the values at the two extremities to 0
    peak_counts_valid[:(nb_neighbor+1)] = 0
    peak_counts_valid[-(nb_neighbor+1):] = 0

    peak_counts = np.zeros(max_period_tested + 1)
    peak_counts[mask_valid_period] = peak_counts_valid

    nfa_binom = np.zeros(max_period_tested + 1) + nb_periods  # include period=0

    for period, count in enumerate(peak_counts):
        p_value = scipy.stats.binom.sf(count, nb_windows, 1/(nb_neighbor*2+1))
        nfa = p_value * nb_periods
        nfa_binom[period] = nfa
    # 0,1,2,3,4,5...,100,101,102,103   nb_neighbor=2 period=100, max=103
    nfa_binom[:nb_neighbor+2] = nb_periods
    nfa_binom[-(nb_neighbor+1):] = nb_periods

    return nfa_binom
