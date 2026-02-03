import numpy as np
import torch


EPS = 1e-17


def integral_image(u:np.ndarray) -> np.ndarray:
    """ Integral image computation

    Parameters
    ----------
    u : np.ndarray
        original image, of shape (H, W)

    Returns
    -------
    np.ndarray
        integral image, of shape (H+1, W+1)
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


class CorrHelper(object):  # using torch for acceleration, faster on CPU than on GPU
    def __init__(self, height:int, width:int, fourier_expanded:np.ndarray, integral_fourier:np.ndarray, 
                 integral_fourier_squared:np.ndarray, window_ratio:float, max_distance:int) -> None:
        """ Helper class to compute correlations between patches at different distances.
            FFT-based acceleration is used to compute correlations between patches.

        Parameters
        ----------
        height : int
            The height of the image
        width : int
            The width of the image
        fourier_expanded : np.ndarray
            The Fourier transform of the image replicated twice in height and in width
        integral_fourier : np.ndarray
            The integral image of the Fourier transform of the image replicated twice in height and in width
        integral_fourier_squared : np.ndarray
            The integral image of the conjugate square Fourier transform of the image replicated twice in height and in width
        window_ratio : float
            The ratio of patch window for a contrario detection
        max_distance : int
            The maximum distance (included) for detecting abnormal correlation
        """

        self.height = height
        self.width = width

        self.integral_fourier = torch.from_numpy(integral_fourier)
        self.integral_fourier_squared = torch.from_numpy(integral_fourier_squared)

        self.patch_height = round(window_ratio * height)
        self.patch_width = round(window_ratio * width)

        patch_offsets = []

        for row_start in np.arange(0, height - self.patch_height + 1, self.patch_height):
            for col_start in np.arange(0, width - self.patch_width + 1, self.patch_width):
                row_start = row_start % height
                col_start = col_start % width
                patch_offsets.append((row_start, col_start))

        patch_offsets = torch.tensor(patch_offsets, dtype=torch.int64)
        self.row_starts = patch_offsets[:, 0]
        self.col_starts = patch_offsets[:, 1]

        # Compute the sum of each patch using the integral image
        self.patch_sums = (
              self.integral_fourier[self.row_starts + self.patch_height, self.col_starts + self.patch_width]
            + self.integral_fourier[self.row_starts, self.col_starts]
            - self.integral_fourier[self.row_starts + self.patch_height, self.col_starts]
            - self.integral_fourier[self.row_starts, self.col_starts + self.patch_width]
        )

        # Compute the sum of the square of each patch using the integral image
        self.patch_square_sums = (
              self.integral_fourier_squared[self.row_starts + self.patch_height, self.col_starts + self.patch_width]
            + self.integral_fourier_squared[self.row_starts, self.col_starts]
            - self.integral_fourier_squared[self.row_starts + self.patch_height, self.col_starts]
            - self.integral_fourier_squared[self.row_starts, self.col_starts + self.patch_width]
        )

        self.patch_norms = torch.sqrt(
            self.patch_square_sums - self.patch_sums * torch.conj(self.patch_sums) / (self.patch_height * self.patch_width)
        )

        top = 0
        self.num_patches_row = (height - self.patch_height) // self.patch_height + 1
        bottom = self.num_patches_row * self.patch_height + top

        left = 0
        self.num_patches_col = (width - self.patch_width) // self.patch_width + 1
        right = self.num_patches_col * self.patch_width + left

        self.fourier_expanded = torch.from_numpy(fourier_expanded[:, left:right])
        self.fourier_cropped = torch.from_numpy(fourier_expanded[top:bottom, left:right])

        # Precompute correlations using FFT at all distances up to max_distance
        self.precomputed_correlations = torch.zeros(
            (max_distance + 1, self.num_patches_row, self.num_patches_col), dtype=torch.complex128
        )

        for col_idx in range(self.num_patches_col):
            img_col = self.fourier_expanded[0:self.height, col_idx * self.patch_width:(col_idx + 1) * self.patch_width]
            img_col_fft = torch.fft.fft(img_col, dim=0, n=self.height)

            for row_idx in range(self.num_patches_row):
                patch = self.fourier_cropped[row_idx * self.patch_height:(row_idx + 1) * self.patch_height, col_idx * self.patch_width:(col_idx + 1) * self.patch_width]
                patch_fft = torch.fft.fft(patch, dim=0, n=self.height)

                # Translate the Fourier transform of the column image
                img_col_fft_translated = img_col_fft * torch.exp(
                    2j * torch.pi * torch.arange(self.height) * row_idx * self.patch_height / (self.height)
                )[:, None]

                correlation = torch.fft.ifft(patch_fft.conj() * img_col_fft_translated, dim=0, n=self.height).sum(dim=1).conj()
                self.precomputed_correlations[:, row_idx, col_idx] = correlation[:max_distance + 1]

        self.precomputed_correlations = self.precomputed_correlations.view(max_distance + 1, self.num_patches_row * self.num_patches_col)

    def compute_correlations(self, distance):
        """ Compute the correlations for all patches at a given distance

        Parameters
        ----------
        distance : int
            The distance to compute correlations

        Returns
        -------
        torch.Tensor
            The correlations for all patches at the given distance, of shape (num_patches,)
        """

        # at distance=0, the correlations are always 1
        if distance == 0:
            return torch.ones(len(self.row_starts))

        row_starts = torch.remainder(self.row_starts + distance, self.height) 
        col_starts = torch.remainder(self.col_starts, self.width)

        # optimization: integral image technique
        patch_sums_2 = \
              self.integral_fourier[row_starts+self.patch_height, col_starts+self.patch_width] \
            + self.integral_fourier[row_starts, col_starts] \
            - self.integral_fourier[row_starts+self.patch_height, col_starts] \
            - self.integral_fourier[row_starts, col_starts+self.patch_width]

        # optimization: integral image technique
        patch_square_sums_2 = \
              self.integral_fourier_squared[row_starts+self.patch_height, col_starts+self.patch_width] \
            + self.integral_fourier_squared[row_starts, col_starts] \
            - self.integral_fourier_squared[row_starts+self.patch_height, col_starts] \
            - self.integral_fourier_squared[row_starts, col_starts+self.patch_width]

        # norm = sqrt(sum of square - sum * conj(sum) / (patch_height * patch_width))
        patch_norms_2 = torch.sqrt(patch_square_sums_2 - patch_sums_2 * torch.conj(patch_sums_2) / (self.patch_height*self.patch_width))

        sum_prod_12_arr = self.precomputed_correlations[distance]

        corrs = torch.abs(
                (sum_prod_12_arr - self.patch_sums * torch.conj(patch_sums_2) / (self.patch_height*self.patch_width)) / (self.patch_norms+EPS) / (patch_norms_2+EPS)
            )

        return corrs


def compute_corr_distances(img_ch, window_ratio, max_distance):
    """ Compute correlations between patches at different distances for one channel of an image

    Parameters
    ----------
    img_ch : np.ndarray
        Image in one channel, of shape (H, W)
    window_ratio : float
        The ratio of patch window w.r.t. image size
    max_distance : int
        The maximum distance (included) for detecting abnormal correlation
    
    Returns
    -------
    np.ndarray
        Correlations for all distances and all anchor patches from 0 to max_distance, of shape (max_distance+1, nb_windows)
    """

    assert len(img_ch.shape) == 2, "img_ch should be a 2D array, got shape {}".format(img_ch.shape)

    h, w = img_ch.shape[0:2]
    
    u_f = np.fft.fftn(img_ch, axes=(0,1), norm="ortho")

    uf_expand = np.tile(u_f, (2,2))

    uf_expand_int = integral_image(uf_expand)

    uf2_expand_int = integral_image(uf_expand * np.conjugate(uf_expand)).real
    
    helper = CorrHelper(h, w, uf_expand, uf_expand_int, uf2_expand_int, window_ratio, max_distance)

    corr_distances = []
    for distance in range(0, max_distance+1):
        corrs = helper.compute_correlations(distance=distance)
        corr_distances.append(corrs)
    
    corr_distances = torch.stack(corr_distances)
    corr_distances = corr_distances.cpu().numpy()

    return corr_distances  # (max_distance+1, nb_windows)
