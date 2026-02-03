import numpy as np
from numba import jit
import scipy
from skimage.util import view_as_windows


@jit(nopython=True)
def _assemble(blocks:np.ndarray) -> np.ndarray:
    nh, nw, block_sz = blocks.shape[0:3]
    h = nh + block_sz - 1
    w = nw + block_sz - 1
    out = np.zeros((h, w))
    denom = np.zeros((h, w))
    for i in range(nh):
        for j in range(nw):
            out[i:i+block_sz, j:j+block_sz] += blocks[i,j]
            denom[i:i+block_sz, j:j+block_sz] += 1
    out = out / denom
    return out


def dct_extract_noise(img:np.ndarray, sigma:float, block_sz:int) -> np.ndarray:
    """ Extract noise using DCT denoiser.

        See ''DCT Image Denoising: a Simple and Effective Image Denoising Algorithm'', Guoshen Yu, and Guillermo Sapiro, https://www.ipol.im/pub/art/2011/ys-dct/
    
    Parameters
    ----------
    img : np.ndarray
        Input image, of shape (H, W).
    sigma : float
        Estimated noise standard deviation.
    block_sz : int
        Block size for DCT denoising.
    
    Returns
    -------
    out : np.ndarray
        Extracted noise, of shape (H, W).
    """

    assert len(img.shape) == 2

    img = img.astype(np.float32)
    blocks = view_as_windows(img, (block_sz, block_sz), (1, 1))
    blocks = scipy.fft.dctn(blocks, axes=(-1,-2), norm="ortho")
    blocks = np.where(np.abs(blocks) < sigma * 3, blocks, np.float32(0))
    blocks = scipy.fft.idctn(blocks, axes=(-1,-2), norm="ortho")
    
    out = _assemble(blocks)

    return out


def test_dct_extract_noise():

    import numpy as np
    img = np.random.normal(0, 10, (256, 256)) + 128
    out = dct_extract_noise(img, sigma=10, block_sz=8)
    print(out)
    print("=== single threading OK === \n")

    from joblib import Parallel, delayed
    import sys
    sys.path.append("..")

    results = Parallel(n_jobs=-1)(delayed(dct_extract_noise)(img, 10, 8) for _ in range(8))
    print("=== multi threading OK === \n")


if __name__ == "__main__":
    test_dct_extract_noise()