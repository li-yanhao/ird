import os
import glob
import argparse

import cv2
import numpy as np
import skimage.filters
import torch
import skimage.io
from PIL import Image
from io import BytesIO

from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append("../")
# from thread_pool_plus import ThreadPoolPlus
from src.resize_kernel_stretch import imresize


WORKERS = 6

def rgb2luminance(img):
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def center_crop(img:np.ndarray, crop_size:int) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    h_center, w_center = h_orig // 2, w_orig // 2
    y1 = h_center - crop_size // 2
    x1 = w_center - crop_size // 2
    x1, y1 = max(0, x1), max(0, y1)
    y2 = y1 + crop_size
    x2 = x1 + crop_size
    return img[y1:y2, x1:x2]


def ups_2d(x, z=None, size=None, mode='bicubic', antialias=False):
    assert z is not None or size is not None

    x = torch.from_numpy(x)[None, None, ...]
    if size is not None:
        x = torch.nn.functional.interpolate(x, size=size, mode=mode, antialias=antialias)
    else:
        x = torch.nn.functional.interpolate(x, scale_factor=z, mode=mode, antialias=antialias)
    x = x[0,0,:,:].cpu().numpy()
    return x


def ups_pil(x:np.ndarray, z=None, size=None, mode="bicubic"):
    assert len(x.shape) == 2
    x_pil = Image.fromarray(x)

    if mode == "bicubic":
        resample_opt = Image.Resampling.BICUBIC
    elif mode == "bilinear":
        resample_opt = Image.Resampling.BILINEAR
    elif mode == "nearest":
        resample_opt = Image.Resampling.NEAREST
    elif mode == "lanczos":
        resample_opt = Image.Resampling.LANCZOS
    
    if size is None:
        h, w = x.shape
        size = (int(h * z), int(w * z))
    x_pil = x_pil.resize(size, resample=resample_opt)

    return np.array(x_pil)


def ups_opencv(x:np.ndarray, z=None, size=None, mode="lanczos"):
    
    assert len(x.shape) == 2

    if mode == "lanczos":
        resample_opt = cv2.INTER_LANCZOS4
    else:
        raise NotImplementedError()
    
    if size is None:
        h, w = x.shape
        size = (int(h * z), int(w * z))

    x = cv2.resize(x, size, interpolation=resample_opt)

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


def create_one_resampled_image(fname:str, downsample_by_2:bool, 
                               antialias:str, antialias_sigma:float,
                               ratio:float, interp:str, q1:int, q2:int, target_size:int):
    img = cv2.imread(fname)[:,:,::-1] 
    if downsample_by_2:
        img = img[::2, ::2, :]

    img = rgb2luminance(img) # gray channel

    # JPEG before resampling
    img = augment_jpeg(img, quality=q1)
    # img = img.round() # this line should be removed
    
    # JPEG after resampling
    # img = augment_jpeg(img, quality=jpeg_q2)
    if abs(ratio - 1) < 1e-5:
        img_resa = np.array(img)
    else:
        if antialias is None:
            if interp in ["lanczos", "nearest"]:
                img_resa = ups_pil(img_resa, z=ratio, mode=interp)
            else:
                # img_resa = ups_2d(img_resa, z=ratio, mode=interp, antialias=False)
                img_resa = imresize(img, scalar_scale=ratio, antialias=False)

        elif antialias == "gaussian":
            if ratio < 1:
                antialias_sigma = antialias_sigma_abs * np.sqrt((1/ratio)**2-1)
                img = skimage.filters.gaussian(img, sigma=antialias_sigma, preserve_range=True)
            if interp == "lanczos":
                img_resa = ups_pil(img_resa, z=ratio, mode=interp)
            else:
                img_resa = ups_2d(img_resa, z=ratio, mode=interp, antialias=False)

        elif antialias == "kernel_stretch":
            img_resa = imresize(img, scalar_scale=ratio, antialias=True)


    # if antialias == "gaussian":
    #     img = skimage.filters.gaussian(img, sigma=antialias_sigma, preserve_range=True)
    #     # print(f"DEBUG: gaussian filter with sigma={antialias_sigma}")
    # if abs(ratio - 1) > 1e-5:
    #     if interp == "lanczos":
    #         img_resa = ups_pil(img, z=ratio, mode=interp)
    #     else:
    #         img_resa = ups_2d(img, z=ratio, mode=interp, antialias=False)
    # else:
    #     img_resa = np.array(img)

    img_resa = center_crop(img_resa, target_size)

    img_resa = augment_jpeg(img_resa, quality=q2)

    img_resa = np.clip(np.round(img_resa), 0, 255)

    return img_resa


def create_resampled_images(target_size, interp, q2, antialias, antialias_sigma_abs):
    RUCHE = False
    MAC = False
    DAMAN = True

    if RUCHE:
        raw_img_folder = "/gpfs/workdir/liy/datasets/raise"
        downsample_by_2 = True
        output_root = "/gpfs/workdir/liy/ird/test/data"  # ruche
        WORKERS = 20

    elif MAC:
        # raw_img_folder = "/Volumes/HPP900/data/raise_cropped"
        raw_img_folder = "/Users/yli/phd/synthetic_image_detection/hongkong/data/raise_cropped"
        downsample_by_2 = False

        # output_root = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/"
        output_root = "data/resample_detection/"
        WORKERS = 6

    elif DAMAN:
        raw_img_folder = "data/raise_cropped"
        downsample_by_2 = False
        output_root = "data/resample_detection/"
        WORKERS = 32


    # raw_img_folder = "../data/raise"  # ruche
    raw_img_fnames = glob.glob(os.path.join(raw_img_folder, "*.png"))
    raw_img_fnames.sort()

    # if RUCHE:
    #     output_root = "/gpfs/workdir/liy/ird/test/data"  # ruche
    # elif MAC:
    #     # output_root = "/Volumes/HPP900/data/resample_detection"
    #     # output_root = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/"
    #     output_root = "data/resample_detection/"

    # target_size = 600
    # ratios = np.arange(0.6, 1.0+1e-10, 0.05)
    ratios = np.arange(0.6, 1.6+1e-10, 0.1)
    # ratios = [0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05]
    # ratios = [1]

    # interp = "nearest"
    # interp = "bilinear"
    # interp = "bicubic"
    # interp = "lanczos"

    jpeg_q1 = -1
    jpeg_q2 = q2


    # pool = ThreadPoolPlus(workers=WORKERS)

    # for fname, orig_size, resa_size, interp, window_ratio, nb_neighbor, jpeg in exp_param_list:
        # img = skimage.io.imread(fname, as_gray=True)
    print("DEBUG")
    # for ratio in ratios:
    #     if antialias:
    #         folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    #     else:
    #         folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    #     os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    for ratio in ratios:
        if antialias is None:
            folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        elif antialias in ["gaussian", "kernel_stretch"]:
            folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}_{antialias}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    def task(fname, ratios):
    
        # img = cv2.imread(fname)[:,:,1].astype(float)  # green channel

        img = cv2.imread(fname)[:,:,::-1] 
        if downsample_by_2:
            img = skimage.filters.gaussian(img, sigma=0.8*np.sqrt(3), preserve_range=True)
            img = img[::2, ::2, :]  # for ruche images
        img = rgb2luminance(img) # gray channel
        
        # JPEG before resampling
        img_orig = augment_jpeg(img, quality=jpeg_q1)
        # img = img.round()
        
        # JPEG after resampling
        # img = augment_jpeg(img, quality=jpeg_q2)
        for ratio in ratios:
            if abs(ratio - 1) <= 1e-5:
                img_resa = np.array(img_orig)

            else:
                if antialias is None:
                    if interp in ["lanczos" or "nearest"]:
                        img_resa = ups_pil(img_orig, z=ratio, mode=interp)
                    else:
                        # img_resa = ups_2d(img_orig, z=ratio, mode=interp, antialias=False)
                        img_resa = imresize(img_orig, scalar_scale=ratio, antialias=False)

                elif antialias == "gaussian":
                    if ratio < 1:
                        antialias_sigma = antialias_sigma_abs * np.sqrt((1/ratio)**2-1)
                        img_resa = skimage.filters.gaussian(img_orig, sigma=antialias_sigma, preserve_range=True)
                    if interp == "lanczos":
                        img_resa = ups_pil(img_resa, z=ratio, mode=interp)
                    else:
                        img_resa = ups_2d(img_resa, z=ratio, mode=interp, antialias=False)
                        
                elif antialias == "kernel_stretch":  # only support bicubic and bilinear
                    img_resa = imresize(img_orig, scalar_scale=ratio, antialias=True)


            img_resa = center_crop(img_resa, target_size)

            if img_resa.shape != (target_size, target_size):
                continue

            img_resa = np.clip(np.round(img_resa), 0, 255).astype(np.uint8)
            # folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"

            if antialias is None:
                folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            elif antialias in ["gaussian", "kernel_stretch"]:
                folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}_{antialias}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            
            # if antialias:
            #     folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            # else:
            #     folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            if jpeg_q2 == -1:
                out_fname = os.path.join(output_root, folder, os.path.basename(fname))
                skimage.io.imsave(out_fname, img_resa)
                print(f"Saved resampled image at {out_fname}")
            else:
                out_fname = os.path.join(output_root, folder, os.path.basename(fname)[:-4] + ".jpg")
                skimage.io.imsave(out_fname, img_resa, quality=jpeg_q2)
                print(f"Saved resampled image at {out_fname}")


    results = Parallel(n_jobs=WORKERS)(delayed(task)(raw_img_fname, ratios) for raw_img_fname in tqdm(raw_img_fnames))

    # for raw_img_fname in raw_img_fnames:
    #     pool.submit(task, raw_img_fname, ratios)
    
    # while not pool.empty():
    #     pool.pop_result()


def random_from_intervals(intervals:list):
    # intervals: [[min0, max0], [min1,max1], ...]

    total_length = 0
    lengths = []
    for interval in intervals:
        vmin, vmax = interval[0], interval[1]
        if vmin >= vmax:
            raise Exception("Each interval must be in [vmin, vmax]")
        lengths.append(vmax - vmin + total_length)
        total_length += vmax - vmin
    
    sampled_pos = np.random.uniform(0, total_length-1e-5)
    interval_index = np.searchsorted(lengths, sampled_pos, side="left")
    print("sampled_pos", sampled_pos)
    print("interval_index", interval_index)

    vmin, vmax = intervals[interval_index][0], intervals[interval_index][1]
    sampled_value = np.random.uniform(vmin, vmax)

    print("interval_index:", interval_index)
    print("vmin:", vmin)
    print("vmax:", vmax)
    print("sampled_value:", sampled_value)

    return sampled_value
    


def create_resampled_images_random_ratios(target_size, interp, q2, antialias, antialias_sigma_abs):
    RUCHE = False
    MAC = False
    DAMAN = True

    if RUCHE:
        raw_img_folder = "/gpfs/workdir/liy/datasets/raise"
        downsample_by_2 = True
        output_root = "/gpfs/workdir/liy/ird/test/data"  # ruche
        WORKERS = 20

    elif MAC:
        # raw_img_folder = "/Volumes/HPP900/data/raise_cropped"
        raw_img_folder = "/Users/yli/phd/synthetic_image_detection/hongkong/data/raise_cropped"
        downsample_by_2 = False

        # output_root = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/"
        output_root = "data/resample_detection/"
        WORKERS = 6

    elif DAMAN:
        raw_img_folder = "data/raise_cropped"
        downsample_by_2 = False
        output_root = "data/resample_detection/"
        WORKERS = 32


    # raw_img_folder = "../data/raise"  # ruche
    raw_img_fnames = glob.glob(os.path.join(raw_img_folder, "*.png"))
    raw_img_fnames.sort()

    ratio_min = 0.6
    ratio_max = 1.6

    jpeg_q1 = -1
    jpeg_q2 = q2

    resamplings_per_image = 5


    if antialias is None:
        folder = f"target_size_{target_size}/r_random_{ratio_min}_to_{ratio_max}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    elif antialias in ["gaussian", "kernel_stretch"]:
        folder = f"target_size_{target_size}/r_random_{ratio_min}_to_{ratio_max}/{interp}_{antialias}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    os.makedirs(os.path.join(output_root, folder), exist_ok=True)


    def task(fname, ratio_min, ratio_max):
    
        # img = cv2.imread(fname)[:,:,1].astype(float)  # green channel

        img = cv2.imread(fname)[:,:,::-1] 
        if downsample_by_2:
            img = skimage.filters.gaussian(img, sigma=0.8*np.sqrt(3), preserve_range=True)
            img = img[::2, ::2, :]  # for ruche images
        img = rgb2luminance(img) # gray channel
        
        # JPEG before resampling
        img_orig = augment_jpeg(img, quality=jpeg_q1)
        # img = img.round()
        
        # JPEG after resampling
        # img = augment_jpeg(img, quality=jpeg_q2)

        for i in range(resamplings_per_image + 1):
            if i == 0:
                ratio = 1
            else:
                ratio = random_from_intervals([[ratio_min, 0.98], [1.02, ratio_max]])

            if abs(ratio - 1) <= 1e-5:
                img_resa = np.array(img_orig)

            else:
                if antialias is None:
                    if interp == "lanczos":
                        img_resa = ups_pil(img_orig, z=ratio, mode=interp)
                    else:
                        img_resa = ups_2d(img_orig, z=ratio, mode=interp, antialias=False)

                elif antialias == "gaussian":
                    if ratio < 1:
                        antialias_sigma = antialias_sigma_abs * np.sqrt((1/ratio)**2-1)
                        img_resa = skimage.filters.gaussian(img_orig, sigma=antialias_sigma, preserve_range=True)
                    if interp == "lanczos":
                        img_resa = ups_pil(img_resa, z=ratio, mode=interp)
                    else:
                        img_resa = ups_2d(img_resa, z=ratio, mode=interp, antialias=False)
                        
                elif antialias == "kernel_stretch":  # only support bicubic and bilinear
                    img_resa = imresize(img_orig, scalar_scale=ratio)


            img_resa = center_crop(img_resa, target_size)

            # if img_resa.shape != (target_size, target_size):
            #     continue

            img_resa = np.clip(np.round(img_resa), 0, 255).astype(np.uint8)
            # folder = f"target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"

            if antialias is None:
                folder = f"target_size_{target_size}/r_random_{ratio_min}_to_{ratio_max}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            elif antialias in ["gaussian", "kernel_stretch"]:
                folder = f"target_size_{target_size}/r_random_{ratio_min}_to_{ratio_max}/{interp}_{antialias}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
            
            
            if jpeg_q2 == -1:
                out_fname = os.path.join(output_root, folder, os.path.basename(fname)[:-4] + f"_r_{ratio:.2f}.png")
                skimage.io.imsave(out_fname, img_resa)
                print(f"Saved resampled image at {out_fname}")
            else:
                out_fname = os.path.join(output_root, folder, os.path.basename(fname)[:-4] + f"_r_{ratio:.2f}.jpg")
                skimage.io.imsave(out_fname, img_resa, quality=jpeg_q2)
                print(f"Saved resampled image at {out_fname}")


    results = Parallel(n_jobs=WORKERS)(delayed(task)(raw_img_fname, ratio_min, ratio_max) for raw_img_fname in tqdm(raw_img_fnames))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, required=False,
                        help='target image size', default=600)
    parser.add_argument('--interp', type=str, required=False,
                        help='interpolator', default='bicubic')
    parser.add_argument('--q2', type=int, required=False,
                        help='post compression quality', default=80)
    parser.add_argument('--antialias', type=int, required=True,
                        help='antialias flag, 0: no antialiasing; 1: antialiasing with Gaussian blur; 2: antialiasing with kernel stretching', 
                        default=0)
    parser.add_argument('--antialias_sigma_abs', type=float, required=False,
                        help='absolute sigma', default=0.8)
    args = parser.parse_args()

    target_size = args.size
    interp = args.interp
    q2 = args.q2

    # if args.antialias == 0:
    #     antialias = False
    # else:
    #     antialias = True  # type 1

    if args.antialias == 0:
        antialias = None
    elif args.antialias == 1:
        antialias = "gaussian"
    else:
        antialias = "kernel_stretch"

    antialias_sigma_abs = args.antialias_sigma_abs

    create_resampled_images(target_size=target_size, interp=interp, q2=q2, 
                            antialias=antialias, antialias_sigma_abs=antialias_sigma_abs)

    # create_resampled_images_random_ratios(target_size=target_size, interp=interp, q2=q2, 
    #                         antialias=antialias, antialias_sigma_abs=antialias_sigma_abs)

# python create_resampled_images.py --q2 -1 --antialias 0 --size 256