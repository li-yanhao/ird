# This is a implementation without optimization but easier to understand.

import numpy as np
import scipy.stats


# if JPEG is considered, the n/8 periods need to be ignored. They should not be used to validate a peak period, 
# and should be 
def detect_resampling_v0(img_in:np.ndarray, window_size:int, nb_neighbor:int, direction:str):
    """_summary_

    Parameters
    ----------
    img_in : np.ndarray
        _description_
    window_size : int
        _description_
    nb_neighbor : int
        _description_
    direction : str
        _description_

    Returns
    -------
    np.ndarray
        a list of NFAs, the i-th element correspond to the NFA of period=i
    """

    assert direction in ["vertical", "horizontal"]

    # assert len(img.shape) == 2

    # print(img_in.shape)    
    if len(img_in.shape) == 2:
        img_in = img_in[:,:,None]

    h, w, c = img_in.shape
    if direction == "vertical":
        nb_periods = h
        len_other = w
    else:
        nb_periods = w
        len_other = w

    max_period_tested = nb_periods//2 + nb_neighbor + 1
    # orig_sz_arr = np.arange(0, nb_periods//2 + nb_neighbor + 2)

    corr_periods = []


    for ch in range(c):
        img = img_in[:,:,ch]
        if direction == "horizontal":
            img = np.array(img.T)

        h, w = img.shape[0:2]

        img = img - np.mean(img)
        u_f = np.fft.fftn(img, axes=(0,1), norm="ortho")

        uf_expand = np.tile(u_f, (2,2)) / (h*w)  # avoid overflow

        uf_expand_int = integral_image(uf_expand)
        uf2_expand_int = integral_image(uf_expand * np.conjugate(uf_expand)).real

        corr_periods_channel = []
        # for i, orig_sz in enumerate(orig_sz_arr):
        for period in range(1, max_period_tested+1):
            # if target_orig_sizes is not None:
            #     ignore = True
            #     for target_orig_sz in target_orig_sizes:
            #         if abs(orig_sz - target_orig_sz) <= 0:
            #             ignore = False
            #     if ignore:
            #         pval_arr.append(1)    
            #         continue

            # lr = window_size
            # lc = window_size

            corr_H1 = swc_disjoint_vertical(
                uf_expand, uf_expand_int, uf2_expand_int,
                h, w, period=period, lr=window_size, lc=window_size)

            corr_periods_channel.append(corr_H1)
    
        corr_periods_channel = np.stack(corr_periods_channel) # (max_period_tested, nb_windows)
        corr_periods.append(corr_periods_channel)
    
    corr_periods = np.concatenate(corr_periods, axis=1) # nb_periods, nb_windows
    # print("corr_periods.shape")
    # print(corr_periods.shape)

    nb_windows = corr_periods.shape[1]

    # print("corr_periods.shape", corr_periods.shape)

    peak_mask = _bool_lateral_maxima(corr_periods, axis=0, order=nb_neighbor, lateral='both')
    peak_counts = np.sum(peak_mask, axis=1)

    # print()
    # print("Binomial test:")
    nfa_binom = np.zeros(1+max_period_tested)  # include period=0
    nfa_binom[0] = nb_periods

    for i, count in enumerate(peak_counts):
        p_value = scipy.stats.binom.sf(count, nb_windows, 1/(nb_neighbor*2+1))  # both
        nfa = p_value * nb_periods
        period = i + 1
        # if nfa < 1e-4 and period >= nb_neighbor+2 and period <= max_period_tested-nb_neighbor-1:
        #     print(f"Period:{period}, peaks: {count}/{nb_windows}, NFA:{nfa}")
        nfa_binom[period] = nfa
    # 0,1,2,3,4,5...,100,101,102,103   nb_neighbor=2 period=100, max=103
    nfa_binom[:nb_neighbor+2] = nb_periods
    nfa_binom[-(nb_neighbor+1):] = nb_periods

    return nfa_binom


