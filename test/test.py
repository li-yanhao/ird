

# define yaml setting protocol: ok
# In yaml setting, there should be 
# - hyparameters: ok
# - output fname: ok

# nfa output: output nfa[i] for period=i
# define output protocol

from dataclasses import dataclass
import glob
import os
import pathlib
import subprocess
from typing import Optional, Tuple
import platform
import argparse

import cv2
import numpy as np
import skimage
from skimage.util import view_as_windows
import scipy
import pandas as pd
import yaml
import torch
from io import BytesIO
from PIL import Image
import dataclasses

# from src import detect_resampling_utest


# from dct_denoise import dct_denoise

import sys
sys.path.append("../src")
from ird import detect_resampling
from misc import rgb2luminance
from create_resampled_images import create_one_resampled_image

# from corr_helper import detect_resampling

import filters


from thread_pool_plus import ThreadPoolPlus


import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)







def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def print_error(*text):
    print("\033[1;31m" + " ".join(map(str, text)) + "\033[0;0m")



class SWC_2D(object):
    def __init__(self, h:int, w:int, 
                 uf_expand:np.ndarray, uf_expand_int:np.ndarray, uf2_expand_int:np.ndarray,
                 window_ratio:float, range_neighbor:int) -> None:
        self.h = h
        self.w = w

        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.uf_expand = np.array(uf_expand)
        self.uf_expand_int = np.array(uf_expand_int)
        # self.uf_expand_int = torch.from_numpy(uf_expand_int).to(self.dev)
        self.uf2_expand_int = np.array(uf2_expand_int)
        # self.uf2_expand_int = torch.from_numpy(uf2_expand_int).to(self.dev)
        # self.window_size = window_size

        self.lr = int(window_ratio * h)
        self.lc = int(window_ratio * w)

        border = self.lr // 2
        self.border = border

        self.range_neighbor = range_neighbor

        # self.r_arr = np.arange(-border, self.h-self.lr+1, self.lr)
        # self.c_arr = np.arange(-border, self.w-self.lc+1, self.lc)

        # rr, cc = np.meshgrid(self.r_arr, self.c_arr, indexing="ij")
        # r_ref_arr = np.mod(rr.reshape(-1), self.h) 
        # c_ref_arr = np.mod(cc.reshape(-1), self.w)

        # # offsets = np.array(offsets)
        # # r1_arr = offsets[:, 0]
        # # c1_arr = offsets[:, 1]

        # self.sum_1_arr = self.uf_expand_int[r_ref_arr+self.lr, c_ref_arr+self.lc] + self.uf_expand_int[r_ref_arr, c_ref_arr] \
        #             - self.uf_expand_int[r_ref_arr+self.lr, c_ref_arr] - self.uf_expand_int[r_ref_arr, c_ref_arr+self.lc]

        # self.sum_square_1_arr = self.uf2_expand_int[r_ref_arr+self.lr, c_ref_arr+self.lc] + self.uf2_expand_int[r_ref_arr, c_ref_arr] - self.uf2_expand_int[r_ref_arr+self.lr, c_ref_arr] - self.uf2_expand_int[r_ref_arr, c_ref_arr+self.lc]

        # self.norm_1_arr = torch.sqrt(self.sum_square_1_arr - self.sum_1_arr * torch.conj(self.sum_1_arr) / (self.lr*self.lc))

        # top = h - border
        # self.num_blob_r = (h*2 - top) // self.lr
        # # self.num_blob_r = h // lr
        # bot = self.num_blob_r * self.lr + top
        # left = w - border
        # self.num_blob_c = (w*2 - left) // self.lc
        # # self.num_blob_c = w // lc
        # right = self.num_blob_c * self.lc + left

        # # print("num_blob_r:", self.num_blob_r)
        # # print("num_blob_c:", self.num_blob_c)

        # self.uf_expand_conj = torch.conj(torch.from_numpy(uf_expand[:(2*h+self.lr*2), left:right])).to(self.dev)  # torch

        # uf_cropped_1 = torch.from_numpy(uf_expand[top:bot, left:right]).to(self.dev)
        # self.sum_prod_12_arr_periods = {}
        # periods = np.arange(1, max_period + 1)

        # for period in periods:
        #     top = (self.h + period - border) % self.h
        #     bot = self.num_blob_r * self.lr + top
        #     uf_cropped_conj_2 = self.uf_expand_conj[top:bot, : ]
        #     prod_12_arr = uf_cropped_1 * uf_cropped_conj_2  # 0.0018s  # slowest! to optimize!
        #     sum_prod_12_arr = prod_12_arr.view(self.num_blob_r, self.lr, self.num_blob_c, self.lc).sum(dim=(1,3)).flatten()
        #     self.sum_prod_12_arr_periods[period] = sum_prod_12_arr


    # def compute_correlations(self, period):
    #     border = self.lr // 2
    #     # r_arr = np.arange(-border, self.h-self.lr+1, self.lr)
    #     # c_arr = np.arange(-border, self.w-self.lc+1, self.lc)
    #     rr, cc = np.meshgrid(self.r_arr, self.c_arr, indexing="ij")
    #     r2_arr = np.mod(rr.reshape(-1) + period, self.h) 
    #     c2_arr = np.mod(cc.reshape(-1), self.w)

    #     # lr = self.window_size
    #     # lc = self.window_size

    #     # integral image technique, low precision
    #     sum_2_arr = self.uf_expand_int[r2_arr+self.lr, c2_arr+self.lc] + self.uf_expand_int[r2_arr, c2_arr] \
    #                 - self.uf_expand_int[r2_arr+self.lr, c2_arr] - self.uf_expand_int[r2_arr, c2_arr+self.lc]
        
    #     sum_square_2_arr = self.uf2_expand_int[r2_arr+self.lr, c2_arr+self.lc] + self.uf2_expand_int[r2_arr, c2_arr] - self.uf2_expand_int[r2_arr+self.lr, c2_arr] - self.uf2_expand_int[r2_arr, c2_arr+self.lc]
    #     norm_2_arr = torch.sqrt(sum_square_2_arr - sum_2_arr * torch.conj(sum_2_arr) / (self.lr*self.lc))  # correct

    #     sum_prod_12_arr = self.sum_prod_12_arr_periods[period]

    #     corrs = torch.abs( (sum_prod_12_arr - self.sum_1_arr * torch.conj(sum_2_arr) / (self.lr*self.lc)) / (self.norm_1_arr+EPS) / (norm_2_arr+EPS) )
        
    #     # debug: cosine similarity
    #     # corrs = torch.abs( (sum_prod_12_arr) / (torch.sqrt(self.sum_square_1_arr)+EPS) / ( torch.sqrt(sum_square_2_arr)+EPS) )

    #     return corrs.cpu().numpy()  # total: 0.007


    def compute_correlations_offset_2d(self, per_0:int, per_1:int, 
                                       anti_diag=False) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        per_0 : int
            the tested period along the horizontal axis
        per_1 : int
            the tested priod along the vertical axis
        anti_diag : bool, optional
            whether the correlation is computed between a patch and another patch positioned at its top-right corner (per_0, per_1) coefficients apart. By default: False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            [0] : correlations between pairs of patches at the target shift (per_0, per_1), in shape (nb_patches)
            [1] : correlations between pairs of patches at random shifts around (per_0, per_1), in shape (nb_shift, nb_patches)
        """
        # assert len(spec_expand.shape) == 2

        eps = 1e-14

        corrs_H1 = []
        corrs_H0 = []

        r_1d_arr = np.arange(-self.border, self.h-self.lr+1, self.lr)
        c_1d_arr = np.arange(-self.border, self.w-self.lc+1, self.lc)
        r_ref_arr, c_ref_arr = np.meshgrid(r_1d_arr, c_1d_arr, indexing="ij")
        r_ref_arr = np.mod(r_ref_arr.flatten(), self.h) 
        c_ref_arr = np.mod(c_ref_arr.flatten(), self.w) 

        range_neighbor = self.range_neighbor
        
        # for delta in np.arange(-range_neighbor, range_neighbor+1):
        #     # if abs(delta) == 1: continue
        #     d0 = per_0 - delta
        #     d1 = per_1 + delta

        for d0 in (per_0 + np.arange(-range_neighbor, range_neighbor+1)):
            for d1 in (per_1 + np.arange(-range_neighbor, range_neighbor+1)):
                # among the offsets [-r,r]x[-r,r], exclude the 8 nearest neighbors of the central point
                if abs(d0 - per_0) <= 1 and abs(d1 - per_1) <= 1 and not (d0 == per_0 and d1 == per_1):
                    continue

                if anti_diag:
                    r_tar_arr = np.mod(r_ref_arr - d0, self.h) 
                    c_tar_arr = np.mod(c_ref_arr + d1, self.w)
                else:
                    r_tar_arr = np.mod(r_ref_arr + d0, self.h) 
                    c_tar_arr = np.mod(c_ref_arr + d1, self.w)

                # integral image technique, low precision
                sum_ref_arr = self.uf_expand_int[r_ref_arr+self.lr, c_ref_arr+self.lc] \
                            + self.uf_expand_int[r_ref_arr, c_ref_arr] \
                            - self.uf_expand_int[r_ref_arr+self.lr, c_ref_arr] \
                            - self.uf_expand_int[r_ref_arr, c_ref_arr+self.lc]
                sum_tar_arr = self.uf_expand_int[r_tar_arr+self.lr, c_tar_arr+self.lc] \
                            + self.uf_expand_int[r_tar_arr, c_tar_arr] \
                            - self.uf_expand_int[r_tar_arr+self.lr, c_tar_arr] \
                            - self.uf_expand_int[r_tar_arr, c_tar_arr+self.lc] \

                sum_square_ref_arr = self.uf2_expand_int[r_ref_arr+self.lr, c_ref_arr+self.lc] \
                                + self.uf2_expand_int[r_ref_arr, c_ref_arr] \
                                - self.uf2_expand_int[r_ref_arr+self.lr, c_ref_arr] \
                                - self.uf2_expand_int[r_ref_arr, c_ref_arr+self.lc] \

                sum_square_tar_arr = self.uf2_expand_int[r_tar_arr+self.lr, c_tar_arr+self.lc] \
                            + self.uf2_expand_int[r_tar_arr, c_tar_arr] \
                            - self.uf2_expand_int[r_tar_arr+self.lr, c_tar_arr] \
                            - self.uf2_expand_int[r_tar_arr, c_tar_arr+self.lc] \

                blobs = view_as_windows(self.uf_expand, (self.lr, self.lc), step=1)
                blob_ref_arr = blobs[r_ref_arr, c_ref_arr].reshape(-1, self.lr * self.lc)
                blob_tar_arr = blobs[r_tar_arr, c_tar_arr].reshape(-1, self.lr * self.lc)


                # option 1: pearson coefficient
                norm_ref_arr = np.sqrt(sum_square_ref_arr - sum_ref_arr * np.conjugate(sum_ref_arr) / (self.lr*self.lc))

                norm_tar_arr = np.sqrt(sum_square_tar_arr - sum_tar_arr * np.conjugate(sum_tar_arr) / (self.lr*self.lc))

                corrs = np.abs(
                    (np.sum(blob_ref_arr * np.conjugate(blob_tar_arr), axis=1) \
                    - sum_ref_arr * np.conjugate(sum_tar_arr) / (self.lr*self.lc)) 
                    / (norm_ref_arr+eps) / (norm_tar_arr+eps) 
                )


                # option 2: cosine similarity
                # norm_ref_arr = np.sqrt(sum_square_ref_arr)

                # norm_tar_arr = np.sqrt(sum_square_tar_arr)

                # corrs = np.abs(
                #     (np.sum(blob_ref_arr * np.conjugate(blob_tar_arr), axis=1)) 
                #     / (norm_ref_arr+eps) / (norm_tar_arr+eps) 
                # )

                # print("corrs.shape")
                # print(corrs.shape)

                if d0 == per_0 and d1 == per_1:
                    corrs_H1 = corrs.copy()
                else:
                    corrs_H0.append(corrs)

        corrs_H0 = np.stack(corrs_H0)

        return corrs_H0, corrs_H1




@dataclass
class ExpConfig:
    fname:str
    # exp-setting
    target_size:int
    resa_ratio: float
    interp:str
    jpeg_q1:int
    jpeg_q2:int
    # hyper-param
    window_ratio:float
    nb_neighbor:int
    direction:str
    sigma:float
    rt_size:int
    data_on_the_fly:bool
    downsample_by_2:bool
    # raw_data_folder:str
    split_validation:bool
    val_range:int
    preprocess:str

    # def __hash__(self):
    #     return hash((self.fname, self.target_size, self.resa_ratio, self.interp, self.jpeg_q1, self.jpeg_q2))


def center_crop(img:np.ndarray, crop_size:int) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    h_center, w_center = h_orig // 2, w_orig // 2
    y1 = h_center - crop_size // 2
    x1 = w_center - crop_size // 2
    x1, y1 = max(0, x1), max(0, y1)
    y2 = y1 + crop_size
    x2 = x1 + crop_size
    return img[y1:y2, x1:x2]


def ups_2d(x, z=None, size=None, mode='bicubic'):
    assert z is not None or size is not None

    x = torch.from_numpy(x)[None, None, ...]
    if size is not None:
        x = torch.nn.functional.interpolate(x, size=size, mode=mode, antialias=False)
    else:
        x = torch.nn.functional.interpolate(x, scale_factor=z, mode=mode, antialias=False)
    x = x[0,0,:,:].cpu().numpy()
    return x


def augment_jpeg(img:np.ndarray, quality:int=-1) -> np.ndarray:
    if quality <= 0:
        return img.astype(float)

    img = np.clip(img.round(), 0, 255)
    img = np.uint8(img)
    img_pil = Image.fromarray(img)
    jpeg_bytes = BytesIO()
    img_pil.save(jpeg_bytes, format="JPEG", quality=int(quality))
    img_pil = Image.open(jpeg_bytes)
    
    img = np.array(img_pil).astype(float)
    return img


def erode(x:np.ndarray, rg:int):
    x_ero = x.copy()
    for r in range(1, rg+1):
        x_ero[:-r] = np.minimum(x_ero[:-r], x[r:])
        x_ero[r:] = np.minimum(x_ero[r:], x[:-r])

    return x_ero


def cross_validate(nfa_0:np.ndarray, nfa_1:np.ndarray, rg:int):
    """ Make agreement between two NFAs, a significant detection from one array must be validated by another significant detection from the other array that is close to the first detection.

    Parameters
    ----------
    nfa_0 : np.ndarray
        the first NFA array
    nfa_1 : np.ndarray
        the second NFA array
    rg : int
        the range of the cross validation, such that out(x) = min_r( max( nfa_0(x), nfa_1(x+r) ) )
    """
    assert len(nfa_0.shape) == 1 and len(nfa_1.shape) == 1 and nfa_0.shape == nfa_1.shape

    # erode nfa_1
    nfa_1_ero = erode(nfa_1, rg)

    return np.maximum(nfa_0, nfa_1_ero)


def process_one_image(cfg:ExpConfig):
    nfa_binom = None
    if cfg.data_on_the_fly:

        img = create_one_resampled_image(cfg.fname, downsample_by_2=cfg.downsample_by_2, ratio=cfg.resa_ratio, interp=cfg.interp, q1=cfg.jpeg_q1, q2=cfg.jpeg_q2, target_size=cfg.target_size).astype(float)

        # ### Option: online processing ###
        # img = skimage.io.imread(cfg.fname)
        # # img = cv2.imread(cfg.fname)[:,:,::-1]
        # img = rgb2luminance(img)
        # # JPEG before resampling
        # img = augment_jpeg(img, quality=cfg.jpeg_q1)
        # if abs(cfg.resa_ratio - 1) > 1e-5:
        #     # img = ups_2d(img, size=int(cfg.orig_size*cfg.resa_ratio), mode=cfg.interp)
        #     img = ups_2d(img, z=cfg.resa_ratio, mode=cfg.interp)
        # img = img.round()
        # img = center_crop(img, cfg.target_size)
        # # resampling
        # # JPEG after resampling
        # img = augment_jpeg(img, quality=cfg.jpeg_q2)
        # ### END of Option ###
    else:
        img = skimage.io.imread(cfg.fname).astype(float)

    assert img.shape == (cfg.target_size, cfg.target_size)


    # unicode = str(abs(hash(cfg)) % (10 ** 8))  # avoid conflict for different input images
    # print("before denoising:", img.sum())
    # print("img type:", img.dtype)

    # if cfg.sigma > 0:
        # improvement
        # img += np.random.normal(0, cfg.sigma / 2, img.shape)  # add noise to cover the saturation part
        # img = dct_denoise(img, sigma=cfg.sigma, block_sz=16)
    

    # img = skimage.io.imread(cfg.fname).astype(float)
    if img.shape != (cfg.target_size, cfg.target_size):
        print_error(f"WARNING: {cfg.fname} is not in shape {(cfg.target_size, cfg.target_size):} !")
        return cfg, nfa_binom

    if cfg.jpeg_q2 > 0:
        is_jpeg = True
    else:
        is_jpeg = False

    if cfg.preprocess == "dct":
        preproc_param = {"sigma": cfg.sigma}
    elif cfg.preprocess == "rt":
        preproc_param = {"rt_size": cfg.rt_size}
    else:
        preproc_param = None
    
    if cfg.direction in ["horizontal", "vertical"]:
        nfa_binom = detect_resampling(
            img, preproc=cfg.preprocess, preproc_param=preproc_param, window_ratio=cfg.window_ratio, 
            nb_neighbor=cfg.nb_neighbor, direction=cfg.direction, is_jpeg=is_jpeg)
    elif cfg.direction == "both":
        nfa_binom_h = detect_resampling(
            img, preproc=cfg.preprocess, window_ratio=cfg.window_ratio, 
            nb_neighbor=cfg.nb_neighbor, direction="horizontal", is_jpeg=is_jpeg)
        nfa_binom_v = detect_resampling(
            img, preproc=cfg.preprocess, window_ratio=cfg.window_ratio, 
            nb_neighbor=cfg.nb_neighbor, direction="vertical", is_jpeg=is_jpeg)
        nfa_binom = cross_validate(nfa_binom_h, nfa_binom_v, cfg.val_range)
    else:
        raise Exception("direction must be `vertical`, `horizontal` or `both`, gets " + cfg.direction + " instead")

    return cfg, nfa_binom


def main_experiment(config_fname:str, resa_ratio:float):
    # load yaml setting
    with open(config_fname, 'r') as file:
        config = yaml.safe_load(file)
    print_info(f"Loaded config from `{config_fname}`")

    print(yaml.dump(config, default_flow_style=False))

    output_fname = os.path.join(config["output_path"], config["experiment_name"] + f"_resa_ratio_{resa_ratio:.2f}.csv")

    if os.path.isfile(output_fname):
        print_error(f"ERROR: `{output_fname}` already exists, please rename the experiment.")
        return

    interp_list = config["interpolator"]
    target_size_list = config["target_size"]
    # resa_ratio_list = np.arange(config["resample_ratio"]["min"],
    #                            config["resample_ratio"]["max"] + 1e-5,
    #                            config["resample_ratio"]["step"])
    jpeg_q1 = config["jpeg_quality"]["before"]
    jpeg_q2 = config["jpeg_quality"]["after"]

    window_ratio = config["window_ratio"]
    nb_neighbor = config["nb_neighbor"]
    direction = config["direction"]
    sigma = config["denoise_sigma"]
    rt_size = config["rt_size"]
    data_on_the_fly = config["data_on_the_fly"]
    downsample_by_2 = config["downsample_by_2"]

    split_validation = config["split_validation"]
    val_range = config["val_range"]

    preprocess = config["preprocess"]

    exp_config_list = []

    if ~data_on_the_fly:
        for interp in interp_list:
            for target_size in target_size_list:
                input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_{resa_ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")
                if jpeg_q2 == -1:
                    fname_list = glob.glob(os.path.join(input_folder, "*.png"))
                else:
                    fname_list = glob.glob(os.path.join(input_folder, "*.jpg"))
            fname_list.sort()

            for fname in fname_list:
                cfg = ExpConfig(fname=fname, target_size=target_size, resa_ratio=resa_ratio,
                                interp=interp, jpeg_q1=jpeg_q1, jpeg_q2=jpeg_q2, window_ratio=window_ratio, nb_neighbor=nb_neighbor, direction=direction, sigma=sigma, rt_size=rt_size, 
                                data_on_the_fly=data_on_the_fly, downsample_by_2=downsample_by_2,
                                split_validation=split_validation, val_range=val_range, preprocess=preprocess)
                exp_config_list.append(cfg)
    
    if data_on_the_fly:
        input_folder = config["data_path"]
        fname_list = glob.glob(os.path.join(input_folder, "*.png"))
        fname_list.sort()

        for fname in fname_list:
            for interp in interp_list:
                for target_size in target_size_list:
                    cfg = ExpConfig(fname=fname, target_size=target_size, resa_ratio=resa_ratio,
                                    interp=interp, jpeg_q1=jpeg_q1, jpeg_q2=jpeg_q2, window_ratio=window_ratio, nb_neighbor=nb_neighbor, direction=direction, sigma=sigma, rt_size=rt_size, 
                                    data_on_the_fly=data_on_the_fly, downsample_by_2=downsample_by_2,
                                    split_validation=split_validation, val_range=val_range, preprocess=preprocess)
                    exp_config_list.append(cfg)
    print(len(exp_config_list))

    # prepare data collection
    df_data = {}

    fields = dataclasses.fields(ExpConfig)
    for v in fields:
        df_data[v.name] = []
    df_data["nfa"] = []
    df_data["nfa_min"] = []

    if not torch.cuda.is_available():  # cpu
        if "macOS" in platform.platform():
            num_workers = 1
        elif "node" in platform.node():
            num_workers = 20
        elif "ruche-mem" in platform.node():
            num_workers = 5
        else:
            num_workers = 6

        pool = ThreadPoolPlus(workers=num_workers)

        for cfg in exp_config_list:
            pool.submit(process_one_image, cfg)

        nb_processed = 0
        nb_positive = 0
        max_nfa = 0
        min_nfa = 1

        while not pool.empty():
            config, nfa_binom = pool.pop_result()
            if nfa_binom is None:
                continue
            threshold = 1e-5
            detected_periods = np.argwhere(nfa_binom<threshold)[:, 0]

            nb_processed += 1
            
            nfa_tmp = np.min(nfa_binom)
            min_nfa = min_nfa if min_nfa < nfa_tmp else nfa_tmp
            max_nfa = max_nfa if max_nfa > nfa_tmp else nfa_tmp

            print(config)
            if len(detected_periods) == 0:
                print("no period detected")
            else:
                nb_positive += 1
                for period in detected_periods:
                    nfa = nfa_binom[period]
                    print(f"period:{period}, NFA:{nfa}")
            print()

            for v in dataclasses.fields(config):
                param = v.name
                value = getattr(config, v.name)
                df_data[param].append(value)
            df_data["nfa"].append(nfa_binom)
            df_data["nfa_min"].append(np.min(nfa_binom))

        print_info(f"positive ratio: {nb_positive} / {nb_processed}")
        print_info(f"min nfa: {min_nfa}")
        print_info(f"max nfa: {max_nfa}")
    
    else: # using cuda
        nb_processed = 0
        nb_positive = 0
        max_nfa = 0
        min_nfa = 1
        threshold = 1e-4

        for cfg in exp_config_list:
            config, nfa_binom = process_one_image(cfg)
            if nfa_binom is None:
                continue

            detected_periods = np.argwhere(nfa_binom<threshold)[:, 0]

            nb_processed += 1
            
            nfa_tmp = np.min(nfa_binom)
            min_nfa = min_nfa if min_nfa < nfa_tmp else nfa_tmp
            max_nfa = max_nfa if max_nfa > nfa_tmp else nfa_tmp
            print(config)
            if len(detected_periods) == 0:
                print("no period detected")
            else:
                nb_positive += 1
                for period in detected_periods:
                    nfa = nfa_binom[period]
                    print(f"period:{period}, NFA:{nfa}")
            print()
            for v in dataclasses.fields(config):
                param = v.name
                value = getattr(config, v.name)
                df_data[param].append(value)
            df_data["nfa"].append(nfa_binom)
            df_data["nfa_min"].append(np.min(nfa_binom))

    df = pd.DataFrame(df_data)
    df['resa_ratio'] = df['resa_ratio'].map(lambda x: '%2.2f' % x)
    df['window_ratio'] = df['window_ratio'].map(lambda x: '%2.2f' % x)
    print(df)

    np.set_printoptions(precision=3)
    df.to_csv(output_fname, index=False, float_format='%4.2e')

    print_info(f"Saved results in `{output_fname}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config file in .yaml')
    parser.add_argument('-r', '--r', type=float, required=False,
                        help='resampling rate', default=-1)
    args = parser.parse_args()


    config_file = args.config

    if args.r == -1:
        # ratios = np.arange(0.6, 2.0+1e-5, 0.1)
        ratios = np.concatenate([
            np.arange(1.2, 2.0+1e-5, 0.1), 
            # np.array([0.91, 0.92, 0.93, 0.94, 0.95, 0.96]),
            # np.array([1.04, 1.05, 1.06, 1.07, 1.08, 1.09]),
            ])
    else:
        ratios = [args.r]

    for resa_ratio in ratios:
        main_experiment(config_file, resa_ratio=resa_ratio)

