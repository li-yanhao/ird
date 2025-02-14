

# define yaml setting protocol: ok
# In yaml setting, there should be 
# - hyparameters: ok
# - output fname: ok

# nfa output: output nfa[i] for period=i
# define output protocol

from dataclasses import dataclass
import glob
import os
import platform
import argparse

import numpy as np
import skimage
import pandas as pd
import yaml
import torch
from io import BytesIO
from PIL import Image
import dataclasses

import sys
sys.path.append("../src")
from ird import detect_resampling
from create_resampled_images import create_one_resampled_image
from thread_pool_plus import ThreadPoolPlus

from joblib import Parallel, delayed
from tqdm import tqdm



import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def print_error(*text):
    print("\033[1;31m" + " ".join(map(str, text)) + "\033[0;0m")


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
    antialias:str
    antialias_sigma_abs:float
    # raw_data_folder:str
    # split_validation:bool
    val_range:int
    preprocess:str
    # discrete_ratio:bool


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
        sigma = np.sqrt(1/(cfg.resa_ratio**2) - 1) * cfg.antialias_sigma_abs
        img = create_one_resampled_image(
            cfg.fname, 
            downsample_by_2=cfg.downsample_by_2,
            antialias=cfg.antialias,
            antialias_sigma=sigma,
            ratio=cfg.resa_ratio, 
            interp=cfg.interp, 
            q1=cfg.jpeg_q1, 
            q2=cfg.jpeg_q2, 
            target_size=cfg.target_size).astype(float)
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
            img, preproc=cfg.preprocess, preproc_param=preproc_param, window_ratio=cfg.window_ratio, 
            nb_neighbor=cfg.nb_neighbor, direction="horizontal", is_jpeg=is_jpeg)
        nfa_binom_v = detect_resampling(
            img, preproc=cfg.preprocess, preproc_param=preproc_param, window_ratio=cfg.window_ratio, 
            nb_neighbor=cfg.nb_neighbor, direction="vertical", is_jpeg=is_jpeg)
        nfa_binom = cross_validate(nfa_binom_h, nfa_binom_v, cfg.val_range)
    else:
        raise Exception("direction must be `vertical`, `horizontal` or `both`, gets " + cfg.direction + " instead")

    d = np.argmin(nfa_binom)
    nfa = nfa_binom[d]
    if nfa < 1e-3:
        print(f"Detect d={d}, NFA={nfa}")
    return cfg, nfa_binom


def main_experiment(config_fname:str, resa_ratio:float):
    """_summary_

    Parameters
    ----------
    config_fname : str
        Configuration file name for the experiment
    resa_ratio : float
        N.B. if resa_ratio = -2, take random ratios
    """

    # load yaml setting
    with open(config_fname, 'r') as file:
        config = yaml.safe_load(file)
    print_info(f"Loaded config from `{config_fname}`")

    print(yaml.dump(config, default_flow_style=False))

    os.makedirs(config["output_path"], exist_ok=True)
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
    antialias = config["antialias"]
    antialias_sigma_abs = config["antialias_sigma_abs"]

    # split_validation = config["split_validation"]
    val_range = config["val_range"]

    preprocess = config["preprocess"]

    exp_config_list = []

    if ~data_on_the_fly:
        for interp in interp_list:
            for target_size in target_size_list:
                if antialias == "None":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_{resa_ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")
                elif antialias == "gaussian":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_{resa_ratio:.2f}/{interp}_gaussian/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")
                elif antialias == "kernel_stretch":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_{resa_ratio:.2f}/{interp}_kernel_stretch/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")

                if jpeg_q2 == -1:
                    fname_list = glob.glob(os.path.join(input_folder, "*.png"))
                else:
                    fname_list = glob.glob(os.path.join(input_folder, "*.jpg"))
            fname_list.sort()

            for fname in fname_list:
                cfg = ExpConfig(fname=fname, target_size=target_size, resa_ratio=resa_ratio,
                                interp=interp, jpeg_q1=jpeg_q1, jpeg_q2=jpeg_q2, window_ratio=window_ratio, nb_neighbor=nb_neighbor, direction=direction, sigma=sigma, rt_size=rt_size, 
                                data_on_the_fly=data_on_the_fly, downsample_by_2=downsample_by_2, 
                                antialias=antialias, antialias_sigma_abs=antialias_sigma_abs,
                                # split_validation=split_validation,
                                val_range=val_range, preprocess=preprocess)
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
                                    antialias=antialias, antialias_sigma_abs=antialias_sigma_abs,
                                    # split_validation=split_validation, 
                                    val_range=val_range, preprocess=preprocess)
                    exp_config_list.append(cfg)
    print(len(exp_config_list))

    # prepare data collection
    df_data = {}

    fields = dataclasses.fields(ExpConfig)
    for v in fields:
        df_data[v.name] = []
    df_data["nfa"] = []
    df_data["nfa_min"] = []


    # if not torch.cuda.is_available():
    if True: # using joblib
        from joblib import Parallel, delayed
        from tqdm import tqdm

        if "macOS" in platform.platform():
            num_workers = 10
        elif "node" in platform.node():
            num_workers = 20
        elif "ruche-mem" in platform.node():
            num_workers = 5
        elif "daman" in platform.node():
            num_workers = 24

        results = Parallel(n_jobs=num_workers)(delayed(process_one_image)(cfg) for cfg in tqdm(exp_config_list))

        nb_processed = 0
        nb_positive = 0
        max_nfa = 0
        min_nfa = 1
        threshold = 1e-5
        for config, nfa_binom in results:
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

        print_info(f"positive ratio: {nb_positive} / {nb_processed}")
        print_info(f"min nfa: {min_nfa}")
        print_info(f"max nfa: {max_nfa}")


        # print("results:")
        # print(results)
        # exit(0)

    # if not torch.cuda.is_available():  # cpu
    elif False: # using thread_pool_plus
        if "macOS" in platform.platform():
            num_workers = 3
        elif "node" in platform.node():
            num_workers = 20
        elif "ruche-mem" in platform.node():
            num_workers = 5
        else:
            num_workers = 8

        pool = ThreadPoolPlus(workers=num_workers)
        print_info(f"Using {num_workers} workers for testing ... ")

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


def main_experiment_random_ratio(config_fname:str):
    """_summary_

    Parameters
    ----------
    config_fname : str
        Configuration file name for the experiment
    """

    # load yaml setting
    with open(config_fname, 'r') as file:
        config = yaml.safe_load(file)
    print_info(f"Loaded config from `{config_fname}`")

    print(yaml.dump(config, default_flow_style=False))

    os.makedirs(config["output_path"], exist_ok=True)
    output_fname = os.path.join(config["output_path"], config["experiment_name"] + f"_random_resa_ratio.csv")

    if os.path.isfile(output_fname):
        print_error(f"ERROR: `{output_fname}` already exists, please rename the experiment.")
        return

    interp_list = config["interpolator"]
    target_size_list = config["target_size"]
    jpeg_q1 = config["jpeg_quality"]["before"]
    jpeg_q2 = config["jpeg_quality"]["after"]

    window_ratio = config["window_ratio"]
    nb_neighbor = config["nb_neighbor"]
    direction = config["direction"]
    sigma = config["denoise_sigma"]
    rt_size = config["rt_size"]
    data_on_the_fly = config["data_on_the_fly"]
    downsample_by_2 = config["downsample_by_2"]
    antialias = config["antialias"]
    antialias_sigma_abs = config["antialias_sigma_abs"]

    val_range = config["val_range"]

    preprocess = config["preprocess"]

    exp_config_list = []

    if ~data_on_the_fly:
        for interp in interp_list:
            for target_size in target_size_list:
                if antialias == "None":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_random_0.6_to_1.6/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")
                elif antialias == "gaussian":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_random_0.6_to_1.6/{interp}_gaussian/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")
                elif antialias == "kernel_stretch":
                    input_folder = os.path.join(config["data_path"], f"target_size_{target_size}/r_random_0.6_to_1.6/{interp}_kernel_stretch/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}")

                if jpeg_q2 == -1:
                    fname_list = glob.glob(os.path.join(input_folder, "*.png"))
                else:
                    fname_list = glob.glob(os.path.join(input_folder, "*.jpg"))
            fname_list.sort()

            for fname in fname_list:
                resa_ratio = float(fname[-8:-4])
                cfg = ExpConfig(fname=fname, target_size=target_size, resa_ratio=resa_ratio,
                                interp=interp, jpeg_q1=jpeg_q1, jpeg_q2=jpeg_q2, window_ratio=window_ratio, nb_neighbor=nb_neighbor, direction=direction, sigma=sigma, rt_size=rt_size, 
                                data_on_the_fly=data_on_the_fly, downsample_by_2=downsample_by_2, 
                                antialias=antialias, antialias_sigma_abs=antialias_sigma_abs,
                                # split_validation=split_validation,
                                val_range=val_range, preprocess=preprocess)
                exp_config_list.append(cfg)
    
    print(len(exp_config_list))

    # prepare data collection
    df_data = {}

    fields = dataclasses.fields(ExpConfig)
    for v in fields:
        df_data[v.name] = []
    df_data["nfa"] = []
    df_data["nfa_min"] = []


    if "macOS" in platform.platform():
        num_workers = 10
    elif "node" in platform.node():
        num_workers = 20
    elif "ruche-mem" in platform.node():
        num_workers = 5
    elif "daman" in platform.node():
        num_workers = 24


    results = Parallel(n_jobs=num_workers)(delayed(process_one_image)(cfg) for cfg in tqdm(exp_config_list))

    nb_processed = 0
    nb_positive = 0
    max_nfa = 0
    min_nfa = 1
    threshold = 1e-5
    for config, nfa_binom in results:
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

    print_info(f"positive ratio: {nb_positive} / {nb_processed}")
    print_info(f"min nfa: {min_nfa}")
    print_info(f"max nfa: {max_nfa}")


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
                        help='set resampling rate: -1: take rates in [0.6, 0.7, ..., 1.6]; -2: take random rates in [0.6, 1.6]; else: take the specific rate',
                        default=-1)
    parser.add_argument('--crop', type=int, required=False,
                        help='crop size', default=-1)
    args = parser.parse_args()


    config_file = args.config

    if args.r == -1:
        ratios = np.arange(0.6, 1.6+1e-5, 0.1)
        # ratios = np.concatenate([
            # np.arange(1.2, 2.0+1e-5, 0.1), 
            # np.array([0.91, 0.92, 0.93, 0.94, 0.95, 0.96]),
            # np.array([1.04, 1.05, 1.06, 1.07, 1.08, 1.09]),
            # ])
    else:
        ratios = [args.r]

    for resa_ratio in ratios:
        main_experiment(config_file, resa_ratio=resa_ratio)

    if args.r == -2:
        main_experiment_random_ratio(config_file)


# usage:
# python test.py --config config/compare_with_nn/discrete_ratio.yaml
# python test.py --config config/compare_with_nn/real_ratio.yaml -r -2  # random ratio