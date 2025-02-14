import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import image

import jpeglib
import iio
import skimage
import argparse
from scipy import fft

import sys
sys.path.append("/workdir/bin/src")
sys.path.append("../src")
from ird import detect_resampling
from misc import rgb2luminance, resize, estimate_original_size_jpeg, estimate_original_size_non_jpeg
from classifier import predict_rate_range, Config


LOG_THRESHOLDS = {
    "rt": -5,
    "tv": -7,
    "dct": -19,
    "phot": -12,
    "none": -17,
}


def filter_by_nms(nfa_histo, threshold, jpeg_periods, max_period):
    positive_mask = nfa_histo < threshold
    mask_jpeg = np.zeros_like(positive_mask, dtype=bool)

    for jpeg_period in jpeg_periods:
        p_left = jpeg_period - 5
        mask_jpeg[p_left+1:jpeg_period+1] = True 
        while p_left >= 0 and positive_mask[p_left] == True:
            mask_jpeg[p_left] = True
            p_left -= 1
        
        p_right = jpeg_period + 5
        mask_jpeg[jpeg_period:p_right] = True
        while p_right < max_period and positive_mask[p_right] == True:
            mask_jpeg[p_right] = True
            p_right += 1

    nfa_histo[mask_jpeg] = 1

    return nfa_histo


def coef2img(dct_coeffs:np.ndarray, q_table:np.ndarray) -> np.ndarray: 
    """ convert DCT 8x8 coefficients to an image

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


def jpeg2luminance(fname:str):
    img_data = jpeglib.read_dct(fname)
    dct_coeffs = img_data.Y
    q_table = img_data.qt[0]
    u0 = coef2img(dct_coeffs, q_table)

    return u0


def center_crop(img:np.ndarray, crop_size:int) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    h_center, w_center = h_orig // 2, w_orig // 2
    y1 = h_center - crop_size // 2
    x1 = w_center - crop_size // 2
    x1, y1 = max(0, x1), max(0, y1)
    y2 = y1 + crop_size
    x2 = x1 + crop_size
    return img[y1:y2, x1:x2]


def main(args):

    print("This method detects whether the image has been resampled, and estimates the possible resampling rates.")
    # print("N.B.: The resampling rate is assumed to be smaller than 2.")
    print()
    print()

    # resize? jpeg?
    # if (args.input.endswith(".jpg") or args.input.endswith(".jpeg")) and args.apply_resize == "false":
    #     img = jpeg2luminance(args.input)
    # else:
    #     img = rgb2luminance(img)

    img = skimage.io.imread(args.input).astype(float)

    if args.apply_resize == "true":
        antialias = (args.antialias == "true")
        img = resize(img, z=args.r, interp=args.interp, antialias=antialias)

    if args.crop > 0:
        # img = img[:args.crop, :args.crop]
        img = center_crop(img, args.crop)

    # iio.write("img_orig.png", img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    
    if args.apply_recompress == "true":
        skimage.io.imsave("img_orig.jpg", img, quality=args.jpegq)
        img_orig_for_ipol = skimage.io.imread("img_orig.jpg")
        skimage.io.imsave("img_orig.png", img_orig_for_ipol)
    else:
        skimage.io.imsave("img_orig.png", img)

    if args.apply_recompress == "true":
        suppress_jpeg = True
    if (args.input.endswith(".jpg") or args.input.endswith(".jpeg")) and args.apply_resize == "false":
        suppress_jpeg = True
    elif args.suppress_jpeg == "true":
        suppress_jpeg = True
    else:
        suppress_jpeg = False


    if args.apply_recompress == "true":
        img = jpeg2luminance("img_orig.jpg")
    else:
        img = rgb2luminance(img)


    # direction = "horizontal"
    if args.direction == "h":
        N = img.shape[1]
        direction = "horizontal"
        # max_period = img.shape[1]
    if args.direction == "v":
        N = img.shape[0]
        direction = "vertical"
        # max_period = img.shape[0]

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
    nfa, img_preproc = detect_resampling(
            img, preproc=preproc, preproc_param=preproc_param, window_ratio=window_ratio, 
            nb_neighbor=nb_neighbor, direction=direction, suppress_jpeg=suppress_jpeg, max_period=N, return_preproc=True)


    # save preprocessed image
    image.imsave("img_preproc.png", img_preproc, cmap='gray')

    # save spec
    spec = np.fft.fft2(img_preproc, norm="ortho", axes=(0,1))
    image.imsave("spec_preproc.png", np.log(np.abs(spec) + 1))

    lognfa = np.log10(nfa + 1e-90)
    log_threshold = LOG_THRESHOLDS[args.preproc]

    if suppress_jpeg:
        lognfa = filter_by_nms(lognfa, log_threshold, np.round(np.arange(1,8)*N/8).astype(int), max_period=N)

    orig_sz_arr = np.arange(0, len(lognfa))

    pairs_d_nfa = []

    fig = plt.figure()
    plt.plot(orig_sz_arr, lognfa, "-", label="nfa")
    for i in range(len(orig_sz_arr)):
        d = orig_sz_arr[i]
        if lognfa[i] < log_threshold:
            pairs_d_nfa.append((i, float(nfa[i])))
            if lognfa[i+1] < -3:
                plt.text(d - 18, lognfa[i], f"{d}", c='r')
            else:
                plt.text(d + 3, lognfa[i], f"{d}", c='r')
            plt.scatter(d, lognfa[i], s=10, color="r")
    # fig.suptitle(fname + "\n" + f"preprocess: {PREPROCESS}", fontsize=10)
    plt.ylabel(r'$log_{10} \, NFA(d)$', fontsize=12)
    plt.xlabel("d", fontsize=12)

    # plt.show()
    plt.savefig("ipol_result.png")

    # if len(pairs_d_nfa) == 0:
    #     print("No resampling traces detected.")
    # else:
    #     print("Detected resampling traces as the abnormal spectral correlation at certain distances.")
        # print("Detected resampling traces as the abnormal spectral correlation at certain distances:")
        # for d, nfa in pairs_d_nfa:
        #     print(f"NFA at distance {d}: {nfa:.2E}")


    # auxiliary module to invert the original M
    pairs_d_nfa = [ (d,nfa) if d <= N/2 else (N-d,nfa) for d,nfa in  pairs_d_nfa]

    # merge close values
    pairs_d_nfa.sort(key=lambda x:x[0])
    pairs_d_nfa = merge_pairs_d_nfa(pairs_d_nfa, epsilon=2)


    # TODO: make the workflow
    ranges, logits = predict_rate_range(img, config=Config())  # already sorted by logits
    # print("logits:", logits)
    # print("ranges:", ranges)
    if logits[0] > 0 and logits[1] < 0: # only one possible range
        ranges = [ranges[0]]
    else:
        ranges = [ranges[0], ranges[1]]

    d_list = [d for d,_ in pairs_d_nfa]
    # print("d_list:", d_list)
    M_list = []
    if len(d_list) > 2:  # the image was likely to be a JPEG image
        for rate_range in ranges:
            M_list.extend(estimate_original_size_jpeg(d_list, N, rate_range, eps=2))
    else:
        for rate_range in ranges:
            M_list.extend(estimate_original_size_non_jpeg(d_list[:2], N, rate_range, eps=2))


    # pairs_d_nfa.sort(key=lambda x:x[1])
    # pairs_d_nfa = pairs_d_nfa[:1]

    # # N = img.shape[1]
    # M_candidates = []
    # for d, nfa in pairs_d_nfa:
    #     M_candidates.append(d)
    #     M_candidates.append(N - d)
    #     M_candidates.append(N + d)
    #     M_candidates.append(2*N - d)
                
    # M_candidates.sort()

    # rate_list = [ N/M for M in M_list]

    # explain results
    print("The method is run successfully in {:.2f} s.".format(time.time() - start))
    print()
    # if len(M_jpeg_list) > 0:
    #     M_jpeg_set = set(M_jpeg_list)
    #     print(f"If the original image is JPEG-compressed, the estimated original sizes M and the estimated resampling" +
    #             f" rates are:")
    #     for M_jpeg in M_jpeg_set:
    #         print(f"  M={M_jpeg:d}, r={N/M_jpeg:.2f}")
    print_results(N, M_list)


def merge_pairs_d_nfa(pairs:list, epsilon=2):
    if not pairs:
        return []

    pairs.sort(key=lambda x:x[0])
    merged_pairs = [pairs[0]]

    for pair in pairs[1:]:
        if abs(pair[0] - merged_pairs[-1][0]) < epsilon:
            if pair[1] < merged_pairs[-1][1]:
                merged_pairs[-1] = pair
        else:
            merged_pairs.append(pair)

    return merged_pairs


def merge_close_values(values:list, epsilon=1e-3):
    """
    Merge values in a list that are very close to each other.

    Args:
    values (list): List of numerical values.
    epsilon (float): Threshold for merging values.

    Returns:
    list: List of merged values.
    """
    if not values:
        return []

    # values.sort()
    merged_values = [values[0]]

    for value in values[1:]:
        if abs(value - merged_values[-1]) < epsilon:
            continue
        else:
            merged_values.append(value)

    return merged_values


def print_results(N:int, M_candidates:list):
    if len(M_candidates) == 0:
        print("No resampling trace is detected.")
        print()
        print("WANING: the following cases are possible:")
        print("- The image is an original image without being resampled;") 
        print("- The image has been severly downsampled so that the resampling traces are undetectable;")
        print("- The image has been resampled but severely post-compressed so that the resampling traces are undetectable.")
    else:
        print("Image resampling traces are detected.")
        print()
        print(f"The tested image having N={N} pixels in width has been resampled from an original image having M pixels in width, with M being one of the following values:")
        for M in M_candidates:
            r = N / M
            print(f"  M={M:d}, resampling rate r=N/M={r:.2f}")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Description of what your script does.")

    # Add required positional arguments
    parser.add_argument("input", type=str, help="Path of the input file.")
    parser.add_argument("--direction", type=str, default="h", choices=["h", "v"])
    parser.add_argument("-p", "--preproc", type=str, default="rt", choices=["rt", "tv", "dct", "phot", "none"])
    parser.add_argument("-r", "--r", type=float, default=1.0)
    parser.add_argument("--apply_resize", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--interp", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--antialias", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--apply_recompress", type=str, default="false", choices=["true", "false"])
    parser.add_argument("-q", "--jpegq", type=int, default=95)
    parser.add_argument("--suppress_jpeg", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--crop", type=int, default=-1)


    # Parse the arguments
    args = parser.parse_args()

    # Pass arguments to the main function
    main(args)

    ## RUN COMMAND
    # python main_ipol.py input_0.png -p $p --apply_resize $apply_resize --r $r --interp $interp 
    # e.g. (local) python main_ipol.py ../test/plot_result/baboon.png -p $p --apply_resize $apply_resize -r $r --interp $interp --antialias $antialias
