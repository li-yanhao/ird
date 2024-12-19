import numpy as np
import torch
from PIL import Image


def rgb2luminance(img:np.ndarray):
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 1:
        return img[:,:,0]
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def resize_torch(x:np.ndarray, size, interp):
    if type(size) is int:
        height, width = size, size
    elif type(size) is tuple:
        assert len(size) == 2, f"`size` should be an integer or an tuple of (height, width), got {size}"
        height, width = size

    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize_torch(x[:,:, c], size, interp)
        return x_re

    x = torch.from_numpy(x)[None, None, ...]
    # print("x", x.shape)
    x_re:torch.Tensor = torch.nn.functional.interpolate(x, size=size, mode=interp, antialias=False)
    x_re = x_re[0,0,:,:].numpy()
    return x_re


def test_resize_torch():
    x = np.random.uniform(0,255, size=(40,40,4)).round()
    x_re = resize_torch(x, 49, interp="bicubic")
    print(x_re.shape)


def resize_pil(x:np.ndarray, size, interp):
    # size: int or tuple (height, width)
    # interp: in ["nearest", "bilinear", "bicubic", "lanczos"]

    if type(size) is int:
        height, width = size, size
    elif type(size) is tuple:
        assert len(size) == 2, f"`size` should be an integer or an tuple of (height, width), got {size}"
        height, width = size

    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize_pil(x[:,:, c], size, interp)
        return x_re

    im = Image.fromarray(x)
    
    if interp == "nearest":
        resample_opt = Image.Resampling.NEAREST
    if interp == "bilinear":
        resample_opt = Image.Resampling.BILINEAR
    if interp == "bicubic":
        resample_opt = Image.Resampling.BICUBIC
    if interp == "lanczos":
        resample_opt = Image.Resampling.LANCZOS
    im_resized = im.resize((width, height), resample=resample_opt)

    return np.array(im_resized)


def test_resize_pil():
    x = np.random.uniform(0,255, size=(40,40)).round()
    x_re = resize_pil(x, 49, interp="lanczos")
    print(x_re)


def resize(x:np.ndarray, z=None, size=None, interp="bicubic", antialias=False):
    assert (z is not None) or (size is not None)

    if size is not None:
        if type(size) is int:
            height, width = size, size
        elif type(size) is tuple:
            assert len(size) == 2
            height, width = size

    elif z is not None:
        height, width = x.shape[0:2]
        height, width = round(height * z), round(width * z)
    
    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize(x[:,:,c], size=(height, width), interp=interp, antialias=antialias)
        return x_re
    
    if antialias:
        x_re = resize_pil(x, size, interp)
    else:
        x_re = resize_torch(x, size, interp)
    return x_re
    


def test_resize():
    x = np.random.uniform(0,255, size=(1000,1000,3)).round()
    for interp in ["nearest", "bilinear", "bicubic", "lanczos"]:
        x_re = resize(x, size=1300, interp=interp, antialias=True)
        x_re = resize(x, size=(1300, 768), interp=interp, antialias=True)
        x_re = resize(x, z=0.76, interp=interp, antialias=True)
        print(x_re.shape)
    for interp in ["nearest", "bilinear", "bicubic"]:
        x_re = resize(x, size=1300, interp=interp, antialias=False)
        x_re = resize(x, size=(1300, 768), interp=interp, antialias=False)
        x_re = resize(x, z=0.76, interp=interp, antialias=False)
        print(x_re.shape)


def ups_2d(x:np.ndarray, z=None, size=None, mode='bicubic', antialias=False):
    """_summary_

    Parameters
    ----------
    x : np.ndarray
        in shape (H, W) or (H, W, C)
    z : _type_, optional
        _description_, by default None
    size : _type_, optional
        _description_, by default None
    mode : str, optional
        _description_, by default 'bicubic'
    antialias : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if np.ndim(x) == 3:
        for c in range(x.shape[2]):
            x[c] = ups_2d(x[..., c], z, size, mode, antialias)
        return x

    assert z is not None or size is not None

    x = torch.from_numpy(x)[None, None, ...]
    if size is not None:
        x = torch.nn.functional.interpolate(x, size=size, mode=mode, antialias=antialias)
    else:
        x = torch.nn.functional.interpolate(x, scale_factor=z, mode=mode, antialias=antialias)
    x = x[0,0,:,:].cpu().numpy()
    return x


if __name__ == "__main__":
    # test_resize_pil()
    # test_resize_torch()
    test_resize()
