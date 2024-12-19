import numpy as np
import cv2
import matplotlib.pyplot as plt
import jpeglib
import iio
import skimage
import argparse

import sys
sys.path.append("../src")
from ird import detect_resampling
from misc import rgb2luminance


def rgb2luminance(img:np.ndarray):
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 1:
        return img[:,:,0]
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def filter_by_nms(nfa_histo, threshold, jpeg_periods, max_period):
    positive_mask = nfa_histo < threshold
    mask_jpeg = np.zeros_like(positive_mask, dtype=bool)

    for jpeg_period in jpeg_periods:
        p_left = jpeg_period - 3
        mask_jpeg[p_left+1:jpeg_period+1] = True 
        while p_left >= 0 and positive_mask[p_left] == True:
            mask_jpeg[p_left] = True
            p_left -= 1
        
        p_right = jpeg_period + 3
        mask_jpeg[jpeg_period:p_right] = True
        while p_right < max_period and positive_mask[p_right] == True:
            mask_jpeg[p_right] = True
            p_right += 1

    nfa_histo[mask_jpeg] = 1

    return nfa_histo



def test_one_image(img_in:np.ndarray, is_jpeg:bool, direction:str="horizontal"):
    assert direction in ["vertical", "horizontal", "joint"]
    assert len(img_in.shape) == 2
    h, w = img_in.shape

    preprocess_list = PREPROCESS.split(",")
        
    img = img_in.copy()
    for proc_type in preprocess_list:
        img = preprocess(img, proc_type=proc_type)

    if cross_val:
        if direction == "vertical":
            nfa_binom_0 = detect_resampling(
                img[:, :img.shape[1]//2], window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR, direction=direction, is_jpeg=is_jpeg)
            nfa_binom_1 = detect_resampling(
                img[:, img.shape[1]//2:], window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR, direction=direction, is_jpeg=is_jpeg)
        elif direction == "horizontal":
            nfa_binom_0 = detect_resampling(
                img[:img.shape[0]//2, :], window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR, direction=direction, is_jpeg=is_jpeg)
            nfa_binom_1 = detect_resampling(
                img[img.shape[0]//2:, :], window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR, direction=direction, is_jpeg=is_jpeg)
        else:
            raise Exception("direction must be `vertical` or `horizontal`, gets " + direction + " instead")
        # nfa_binom = np.maximum(nfa_binom_left, nfa_binom_right)
        nfa_binom = nfa_binom_0.copy()
        nfa_binom[1:-1] = np.maximum(
            nfa_binom_0[1:-1],
            np.minimum(nfa_binom_1[0:-2], nfa_binom_1[1:-1], nfa_binom_1[2:])
        )
    else:
        nfa_binom = detect_resampling(img, window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR, direction=direction, is_jpeg=is_jpeg)

    return nfa_binom, img


def test_one_file(fname:str):
    if fname.endswith(".jpg"):
        is_jpeg = True
    else:
        is_jpeg = False

    if is_jpeg:
        img = jpeglib.read_spatial(fname, out_color_space=jpeglib.JCS_GRAYSCALE).spatial[:,:,0]
        print(img.shape)
    else:
        img = cv2.imread(fname)[:,:,::-1]


    if "deblock" in PREPROCESS:
        img = filters.tv_deblock(fname)


    if len(img.shape) == 3:
        img = rgb2luminance(img)


    nfa_list_hori, residual = test_one_image(img, is_jpeg=is_jpeg, direction="horizontal")
    nfa_list_vert, residual = test_one_image(img, is_jpeg=is_jpeg, direction="vertical")

    apply_nms = True
    threshold = 1e-3 # todo: set this for different JPEG quality
    if apply_nms:
        jpeg_periods = np.arange(1, 8) * 75
        nfa_list_hori = filter_by_nms(nfa_list_hori, threshold, jpeg_periods=jpeg_periods, max_period=600)
        nfa_list_vert = filter_by_nms(nfa_list_vert, threshold, jpeg_periods=jpeg_periods, max_period=600)

    lognfa_list_hori = np.log10(nfa_list_hori)
    lognfa_list_vert = np.log10(nfa_list_vert)

    orig_sz_arr = np.arange(0, len(lognfa_list_hori))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 5)
    fig.suptitle(fname + "\n" + f"preprocess: {PREPROCESS}", fontsize=10)

                
    ax0 = fig.add_subplot(gs[:, :2])
    ax1 = fig.add_subplot(gs[0, 2:])
    ax2 = fig.add_subplot(gs[1, 2:])

    # ax0.imshow(img, cmap="gray")
    ax0.imshow(residual, cmap="gray")

    ax1.plot(orig_sz_arr, lognfa_list_hori, "-", label="nfa")
    ax1.set_ylabel(r'$log_{10} \, NFA(d)$', fontsize=12)
    ax1.set_xlabel("d", fontsize=12)
    ax1.set_title("horizontal detection")
    for i in range(len(orig_sz_arr)):
        period = orig_sz_arr[i]
        if lognfa_list_hori[i] < -3:
            if lognfa_list_hori[i+1] < -3:
                ax1.text(period - 18, lognfa_list_hori[i], f"{period}", c='r')
            else:
                ax1.text(period + 3, lognfa_list_hori[i], f"{period}", c='r')
            ax1.scatter(period, lognfa_list_hori[i], s=10, color="r")

    orig_sz_arr = np.arange(0, len(lognfa_list_vert))

    ax2.plot(orig_sz_arr, lognfa_list_vert, "-", label="nfa")
    ax2.set_ylabel(r'$log_{10} \, NFA(d)$', fontsize=12)
    ax2.set_xlabel("d", fontsize=12)
    ax2.set_title("vertical detection")
    for i in range(len(orig_sz_arr)):
        period = orig_sz_arr[i]
        if lognfa_list_vert[i] < -3:
            if lognfa_list_vert[i+1] < -3:
                ax2.text(period - 18, lognfa_list_vert[i], f"{period}", c='r')
            else:
                ax2.text(period + 3, lognfa_list_vert[i], f"{period}", c='r')
            ax2.scatter(period, lognfa_list_vert[i], s=10, color="r")

    plt.tight_layout()
    plt.show()


def main(args):

    # img = skimage.io.imread(args.input).astype(float)
    img = iio.read(args.input)
    img = rgb2luminance(img)

    if args.crop is not None:
        x, y, w, h = tuple(args.crop)
        img = img[y:y+h, x:x+w]

    if args.input.endswith(".jpg") or args.input.endswith(".jpeg"):
        is_jpeg = True
    else:
        is_jpeg = False

    if args.direction == "h":
        direction = "horizontal"
        max_period = img.shape[1]
    if args.direction == "v":
        direction = "vertical"
        max_period = img.shape[0]

    preproc_param = None
    if args.preproc == "rt":
        preproc = "rt"
        preproc_param = {"rt_size": 3}
    if args.preproc == "tv":
        preproc = "tv"
    if args.preproc == "dct":
        preproc = "dct"
        preproc_param = {"sigma": 3}
    if args.preproc == "phot":
        preproc = "phot"
    if args.preproc == "none":
        preproc = "none"
    
    window_ratio = 0.1
    nb_neighbor = 20
    
    import time
    start = time.time()
    nfa = detect_resampling(
            img, preproc=preproc, preproc_param=preproc_param, window_ratio=window_ratio, 
            nb_neighbor=nb_neighbor, direction=direction, is_jpeg=is_jpeg, max_period=max_period)
    print("Spent time:", time.time() - start)

    # print("nfa")
    # print(nfa)


    lognfa = np.log10(nfa + 1e-10)
    orig_sz_arr = np.arange(0, len(lognfa))

    # visualize the results
    # fig = plt.figure(figsize=(16, 8))
    fig = plt.figure()
    plt.plot(orig_sz_arr, lognfa, "-", label="nfa")
    for i in range(len(orig_sz_arr)):
        d = orig_sz_arr[i]
        if lognfa[i] < -3:
            if lognfa[i+1] < -3:
                plt.text(d - 18, lognfa[i], f"{d}", c='r')
            else:
                plt.text(d + 3, lognfa[i], f"{d}", c='r')
            plt.scatter(d, lognfa[i], s=10, color="r")
    # fig.suptitle(fname + "\n" + f"preprocess: {PREPROCESS}", fontsize=10)
    plt.ylabel(r'$log_{10} \, NFA(d)$', fontsize=12)
    plt.xlabel("d", fontsize=12)

    plt.show()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Description of what your script does.")

    # Add required positional arguments
    parser.add_argument("input", type=str, help="Path of the input file.")
    parser.add_argument("--direction", type=str, default="h", choices=["h", "v"])
    parser.add_argument("--preproc", type=str, default="rt", choices=["rt", "tv", "dct", "phot", "none"])

    parser.add_argument('--crop', nargs=4, type=int, metavar=("x", "y", "w", "h"), default=None)
    



    # Add optional arguments
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    # parser.add_argument("-o", "--optional_value", type=int, default=42,
    #                     help="An optional integer value. Default is 42.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass arguments to the main function
    main(args)