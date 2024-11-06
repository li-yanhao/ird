import numpy as np
from numba import jit
import scipy
from skimage.util import view_as_windows, view_as_blocks


@jit(nopython=True)
def assemble(blocks:np.ndarray): # Function is compiled and runs in machine code
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
    # out = out / (denom + 1e-10)
    return out


def dct_extract_noise(img:np.ndarray, sigma:float, block_sz:int):
    """ The output is the noise """
    assert len(img.shape) == 2

    img = img.astype(np.float32)
    blocks = view_as_windows(img, (block_sz, block_sz), (1, 1))
    blocks = scipy.fft.dctn(blocks, axes=(-1,-2), norm="ortho")
    blocks = np.where(np.abs(blocks) < sigma * 3, blocks, np.float32(0))
    blocks = scipy.fft.idctn(blocks, axes=(-1,-2), norm="ortho")
    
    out = assemble(blocks)
    return out




@jit(nopython=True)
def assemble_8x8(blocks:np.ndarray): # Function is compiled and runs in machine code
    nh, nw, block_sz = blocks.shape[0:3]
    h = nh * block_sz
    w = nw * block_sz

    # h = nh + block_sz - 1
    # w = nw + block_sz - 1
    out = np.zeros((h, w))
    for i in range(nh):
        for j in range(nw):
            out[i*block_sz:(i+1)*block_sz, j*block_sz:(j+1)*block_sz] = blocks[i,j]
    out = out
    return out


def dct_extract_noise_8x8(img:np.ndarray, sigma:float):
    """ The output is the noise """
    assert len(img.shape) == 2

    block_sz = 8

    img = img.astype(np.float32)
    blocks = view_as_blocks(img, (block_sz, block_sz))
    blocks = scipy.fft.dctn(blocks, axes=(-1,-2), norm="ortho")
    blocks = np.where(np.abs(blocks) < sigma * 3, blocks, np.float32(0))
    blocks = scipy.fft.idctn(blocks, axes=(-1,-2), norm="ortho")
    
    out = assemble_8x8(blocks)
    return out


def dct_extract_noise_try(img:np.ndarray, block_sz:int):
    """ The output is the noise """
    assert len(img.shape) == 2

    img = img.astype(np.float32)
    blocks = view_as_windows(img, (block_sz, block_sz), (1, 1))
    blocks = scipy.fft.dctn(blocks, axes=(-1,-2), norm="ortho")
    # blocks[:,:,0:block_sz//2,:] = 0
    # blocks[:,:,:,0:block_sz//2] = 0
    energies = np.abs(blocks)
    # probas = 1 / energies
    # probas = probas / np.sum(probas, axis=(-1, -2))[..., None, None]

    energies_prob = energies / np.sum(energies, axis=(-1, -2))[..., None, None]
    probas = 1 / energies_prob  # diminish the energy by randomly canceling some frequencies
    probas = 10 * probas / np.sum(probas, axis=(-1, -2))[..., None, None]  # diminish the energy by randomly canceling some frequencies
    print(probas)
    blocks_mask = np.random.uniform(0, 1, size=blocks.shape) < probas
    print(np.abs(blocks).sum())
    blocks = blocks * blocks_mask
    print(np.abs(blocks).sum())
    
    # blocks = np.where(np.abs(blocks) < sigma * 3, blocks, np.sign(blocks) * sigma * 3)
    blocks = scipy.fft.idctn(blocks, axes=(-1,-2), norm="ortho", workers=8)
    
    out = assemble(blocks)
    return out


# def dct_extract_noise_quantile(img:np.ndarray, quantile:float, block_sz):
#     """ The output is the noise """
#     assert len(img.shape) == 2

#     img = img.astype(np.float32)
#     blocks = view_as_windows(img, (block_sz, block_sz), (1, 1))
#     blocks = scipy.fft.dctn(blocks, axes=(-1,-2), norm="ortho")
#     # blocks = np.where(np.abs(blocks) < sigma * 3, blocks, np.float32(0))
#     blocks_abs = np.abs(blocks)
#     mask = (blocks_abs > np.quantile(blocks_abs, quantile, axis=(-1, -2))[..., None, None])
#     # print(mask.shape)
#     blocks[mask] = 0
#     blocks = scipy.fft.idctn(blocks, axes=(-1,-2), norm="ortho")
    
#     out = assemble(blocks)
#     return out

