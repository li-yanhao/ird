from scipy.fft import fft2, ifft2
import numpy as np

def phase_transform(img:np.ndarray) -> np.ndarray:
    spec = fft2(img, axes=(0, 1), norm="ortho")
    spec /= np.abs(spec)
    return ifft2(spec, axes=(0, 1), norm="ortho").real

