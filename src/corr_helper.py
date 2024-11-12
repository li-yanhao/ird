import numpy as np
import torch
import scipy
from skimage.util import view_as_windows


EPS = 1e-17

def integral_image(u:np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    u : np.ndarray
        original image

    Returns
    -------
    np.ndarray
        integral image
    """

    assert len(u.shape) == 2

    if u.dtype == float:
        v = np.zeros((u.shape[0]+1, u.shape[1]+1), dtype=np.float64)
        v[1:,1:] = u
    elif u.dtype == np.complex128 or u.dtype == np.complex64:
        v = np.zeros((u.shape[0]+1, u.shape[1]+1), dtype=np.complex128)
        v[1:,1:] = u

    v = np.cumsum(v, axis=0)
    v = np.cumsum(v, axis=1)

    return v



class CorrHelper(object):
    def __init__(self, h:int, w:int, uf_expand:np.ndarray, uf_expand_int:np.ndarray, 
                 uf2_expand_int:np.ndarray, window_ratio:float, max_period:int) -> None:
        """ 

        Parameters
        ----------
        h : int
            The height of the image
        w : int
            The width of the image
        uf_expand : np.ndarray
            The fourier transform of the image replicated twice in height and in width
        uf_expand_int : np.ndarray
            The integral image of the fourier transform of the image replicated twice in height and in width
        uf2_expand_int : np.ndarray
            The integral image of the conjuguate square fourier transform of the image replicated twice in height and in width
        window_ratio : float
            The ratio of patch window for a contrario detection
        max_period : int
            The maximum period (included) for detecting abnormal correlation
        """

        self.h = h
        self.w = w

        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.uf_expand_int = torch.from_numpy(uf_expand_int).to(self.dev)
        self.uf2_expand_int = torch.from_numpy(uf2_expand_int).to(self.dev)

        self.lr = round(window_ratio * h)
        self.lc = round(window_ratio * w)

        offsets = []

        # default version
        for r1 in np.arange(0, h-self.lr+1, self.lr):
            for c1 in np.arange(0, w-self.lc+1, self.lc):
                r1 = r1 % h
                c1 = c1 % w
                offsets.append((r1, c1))
        
        offsets = np.array(offsets)
        self.r1_arr = offsets[:, 0]
        self.c1_arr = offsets[:, 1]

        # Optimization: compute the sum of each patch from integral image
        self.sum_1_arr = \
              self.uf_expand_int[self.r1_arr+self.lr, self.c1_arr+self.lc] \
            + self.uf_expand_int[self.r1_arr, self.c1_arr] \
            - self.uf_expand_int[self.r1_arr+self.lr, self.c1_arr] \
            - self.uf_expand_int[self.r1_arr, self.c1_arr+self.lc]

        # Optimization: compute the sum of the square of each patch from integral image
        self.sum_square_1_arr = \
              self.uf2_expand_int[self.r1_arr+self.lr, self.c1_arr+self.lc] \
            + self.uf2_expand_int[self.r1_arr, self.c1_arr] \
            - self.uf2_expand_int[self.r1_arr+self.lr, self.c1_arr] \
            - self.uf2_expand_int[self.r1_arr, self.c1_arr+self.lc]

        self.norm_1_arr = torch.sqrt(self.sum_square_1_arr - self.sum_1_arr * torch.conj(self.sum_1_arr) / (self.lr*self.lc))

        top = 0
        self.num_blob_r = (h-1) // self.lr + 1
        bot = self.num_blob_r * self.lr + top

        left = 0
        self.num_blob_c = (w-1) // self.lc + 1
        right = self.num_blob_c * self.lc + left

        # self.uf_expand_conj = torch.conj(torch.from_numpy(uf_expand[:(2*h+self.lr*2), left:right])).to(self.dev)  # todo: reduce the size
        self.uf_expand_conj = torch.conj(torch.from_numpy(uf_expand[:, left:right])).to(self.dev)  # todo: reduce the size

        self.uf_cropped_1 = torch.from_numpy(uf_expand[top:bot, left:right]).to(self.dev)
        self.sum_prod_12_arr_periods = {}
        periods = np.arange(1, max_period + 1)

        for period in periods:
            top = period % self.h
            bot = self.num_blob_r * self.lr + top
            uf_cropped_conj_2 = self.uf_expand_conj[top:bot, : ]
            prod_12_arr = self.uf_cropped_1 * uf_cropped_conj_2
            sum_prod_12_arr = prod_12_arr.view(self.num_blob_r, self.lr, self.num_blob_c, self.lc).sum(dim=(1,3)).flatten()
            self.sum_prod_12_arr_periods[period] = sum_prod_12_arr


    def compute_correlations(self, period):
        # at period=0, the correlations are always 1
        if period == 0:
            return np.ones(len(self.r1_arr))

        r2_arr = np.mod(self.r1_arr + period, self.h) 
        c2_arr = np.mod(self.c1_arr, self.w)

        # optimization: integral image technique
        sum_2_arr = \
              self.uf_expand_int[r2_arr+self.lr, c2_arr+self.lc] \
            + self.uf_expand_int[r2_arr, c2_arr] \
            - self.uf_expand_int[r2_arr+self.lr, c2_arr] \
            - self.uf_expand_int[r2_arr, c2_arr+self.lc]
        
        # optimization: integral image technique
        sum_square_2_arr = \
              self.uf2_expand_int[r2_arr+self.lr, c2_arr+self.lc] \
            + self.uf2_expand_int[r2_arr, c2_arr] \
            - self.uf2_expand_int[r2_arr+self.lr, c2_arr] \
            - self.uf2_expand_int[r2_arr, c2_arr+self.lc]

        norm_2_arr = torch.sqrt(sum_square_2_arr - sum_2_arr * torch.conj(sum_2_arr) / (self.lr*self.lc))

        sum_prod_12_arr = self.sum_prod_12_arr_periods[period]

        corrs = torch.abs(
                (sum_prod_12_arr - self.sum_1_arr * torch.conj(sum_2_arr) / (self.lr*self.lc)) / (self.norm_1_arr+EPS) / (norm_2_arr+EPS) 
            )
        
        return corrs.cpu().numpy()


def compute_corr_periods(img_ch, window_ratio, max_period):
    h, w = img_ch.shape[0:2]
    
    u_f = np.fft.fftn(img_ch, axes=(0,1), norm="ortho")

    uf_expand = np.tile(u_f, (2,2)) / (h*w)

    uf_expand_int = integral_image(uf_expand)
    uf2_expand_int = integral_image(uf_expand * np.conjugate(uf_expand)).real
    corr_periods = []

    helper = CorrHelper(h, w, uf_expand, uf_expand_int, uf2_expand_int, window_ratio, max_period)

    for period in range(0, max_period+1):
        corrs = helper.compute_correlations(period=period)
        corr_periods.append(corrs)

    corr_periods = np.stack(corr_periods)

    return corr_periods
