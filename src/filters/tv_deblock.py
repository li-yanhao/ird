import jpeglib
import numpy as np
from skimage.util import view_as_blocks
from scipy import fft
import torch



def img2coef(img, q_table:np.ndarray):
    img = img - 128
    blocks = view_as_blocks(img, (8, 8))

    dct_blocks = fft.dctn(blocks, axes=(-1, -2), norm="ortho", workers=4)  # N, 8, 8
    dct_blocks /= q_table

    return dct_blocks


def coef2img(dct_coeffs:np.ndarray, q_table:np.ndarray) -> np.ndarray: 
    """_summary_

    Parameters
    ----------
    dct_blocks : np.ndarray
        In shape (nr, nc, 8, 8)
    q_table : np.ndarray
        In shape (8, 8)

    Returns
    -------
    np.ndarray
        Luminance channel
    """

    nr, nc = dct_coeffs.shape[:2]
    dct_blocks = dct_coeffs * q_table

    restore_blocks = fft.idctn(dct_blocks, axes=(-1, -2), norm="ortho", workers=4)
    img = np.transpose(restore_blocks, (0, 2, 1, 3)).reshape(nr*8, nc*8)

    img = img + 128

    return img


def tv(u: torch.Tensor):
    """ Classical l2 total variation

    Parameters
    ----------
    u : torch.Tensor
        A 2D image, suppose the both sides are multiples of 8
    """
    ua = torch.cat((u[1:, :], u[-1:, :]), dim=0) - u
    ub = torch.cat((u[:, 1:], u[:, -1:]), dim=1) - u
    j_u = torch.sum(torch.sqrt(ua**2 + ub**2 + 1e-8))

    return j_u


def deblock(u0_coef:np.ndarray, q_table:np.ndarray, K:int, tv_type:str) -> np.ndarray:
    """ Deblock an image

    Parameters
    ----------
    u0_coef : np.ndarray
        quantized dct coefficients read from the jpeg file, of shape (nr, nc, 8, 8)
    q_table : np.ndarray
        quantization table, of shape (8, 8)
    K : int
        number of iterations to solve the TV-minimization problem
    tv_type : str
        type of total variation, supported types are ["TV", "sTV", "aTV"]

    Returns
    -------
    np.ndarray
        deblocked image, of shape (nr*8, nc*8)
    """

    u0 = coef2img(u0_coef, q_table)

    u = np.array(u0)
    
    j_opt = tv(torch.from_numpy(u))
    u_opt = u0
    for k in range(K):
        tk = 1 / (k+1)

        u = torch.from_numpy(u)
        u.requires_grad_()
        tv_of_u = tv(u)

        if j_opt > tv_of_u.item():
            j_opt = tv_of_u.item()
            u_opt = u.detach().numpy()

        gu = torch.autograd.grad(outputs=tv_of_u, inputs=u)[0]

        v = u.clone()

        v -= tk * gu

        v = v.detach().numpy()  # h, w

        v_coef = img2coef(v, q_table)  # n, 8, 8

        # project the image in the authorized space of images so that the deblocked image is JPEG compression-consistent
        v_coef = np.clip(v_coef, u0_coef-0.5, u0_coef+0.5)

        P_v = coef2img(v_coef, q_table)

        u = P_v

    return u_opt


def tv_deblock(fname:str):
    # read qt table and coeffs
    
    # img_data = jpeglib.read_dct('wechat.jpg')
    img_data = jpeglib.read_dct(fname)
    # print(im.Y.shape)
    # print(im.qt)

    dct_coeffs = img_data.Y
    q_table = img_data.qt[0]

    img_data = jpeglib.read_dct(fname)
    dct_coeffs = img_data.Y
    u_deblock = deblock(dct_coeffs, q_table, K=10, tv_type="TV")


    return u_deblock
