import numpy as np
from numba import jit


@jit(nopython=True)
def rank_transform(img:np.ndarray, sz:int) -> np.ndarray:
    r""" Compute the rank transform of an image.
    
    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W).
    sz : int
        Half size of the neighborhood. The neighborhood is of size ``(2*sz+1, 2*sz+1)``.
    
    Returns
    -------
    out : np.ndarray, float32
        Rank transformed image of shape (H, W), with values in ``[ 0, (2*sz+1)^2 )``. 
    """

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    anchor = 0
    for i in range(sz, h-sz):
        for j in range(sz, w-sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[i+di, j+dj] < anchor:
                        rank += 1
                    elif img[i+di, j+dj] == anchor:
                        rank += 0.5
            out[i, j] = rank

    return out


def test_rank_transform():
    img = np.arange(9*9).reshape(9, 9).astype(np.float32)
    out = rank_transform(img, sz=1)
    print(img)
    print(out)


if __name__ == "__main__":
    test_rank_transform()

