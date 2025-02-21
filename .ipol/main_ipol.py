import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

import jpeglib
import skimage
import argparse
from scipy import fft

import sys
sys.path.append("/workdir/bin/src")
sys.path.append("../src")
from ird import detect_resampling
from misc import (
    rgb2luminance, resize, 
    estimate_original_size_jpeg, 
    estimate_original_size_non_jpeg, 
    estimate_original_size_by_jpegx16,
    verify_upsample_by_2_pow_n
    )
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

    # print("This method detects whether the image has been resampled, and estimates the possible resampling rates.")
    # print("N.B.: The resampling rate is assumed to be smaller than 2.")
    print()
    print()

    # resize? jpeg?
    # if (args.input.endswith(".jpg") or args.input.endswith(".jpeg")) and args.apply_resize == "false":
    #     img = jpeg2luminance(args.input)
    # else:
    #     img = rgb2luminance(img)

    img = skimage.io.imread(args.input).astype(float)
    if np.ndim(img) == 3:
        img = img[:,:,:3]  # if RGBA, take only RGB

    if args.apply_resize == "true":
        antialias = (args.antialias == "true")

        if args.crop > 0:
            # pre-crop the image to reduce the processing time
            img = center_crop(img, int(args.crop/args.r) + 5)
            
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
    elif (args.input.lower().endswith(".jpg") or args.input.lower().endswith(".jpeg")) and args.apply_resize == "false":
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
    ranges_sorted, logits_sorted = predict_rate_range(img, config=Config())  # already sorted by logits
    # print("logits:", logits)
    # print("ranges:", ranges)
    if logits_sorted[0] > 0 and logits_sorted[1] < 0: # only one possible range
        ranges_jpeg = [ranges_sorted[0]]
    else:
        ranges_jpeg = [ranges_sorted[0], ranges_sorted[1]]

    ranges_non_jpeg = [ranges_sorted[0], ranges_sorted[1], ranges_sorted[2]]

    d_list = [d for d,_ in pairs_d_nfa]

    print("The method is run successfully in {:.2f} s.".format(time.time() - start))
    print()

    M_list = []
    if len(d_list) > 2:  # the image was likely to be a JPEG image
        # print("The original image is likely to be a JPEG image")
        for rate_range in ranges_jpeg:
            print("rate_range", rate_range)
            M_list.extend(estimate_original_size_jpeg(d_list, N, rate_range, eps=2))
    else:
        # print("The original image is unlikely to be a JPEG image")
        for rate_range in ranges_non_jpeg:
            print("rate_range", rate_range)
            M_list.extend(estimate_original_size_non_jpeg(d_list[:2], N, rate_range, eps=2))


    print_results(N, M_list)

    # lowest_nfa =
    # print_results(N, M_list, lowest_nfa)



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


def write_tuples_to_txt(tuples_list, labels_list, filename):
    """Writes a list of tuples with corresponding labels to a text file.
    """
    try:
        with open(filename, 'w') as f:
            f.write(", ".join(labels_list))
            f.write("\n")
            for tup in tuples_list:
                f.write(", ".join(map(str, tup)))
                f.write("\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def main_bidirection(args):
    img = skimage.io.imread(args.input).astype(float)
    if np.ndim(img) == 3:
        img = img[:,:,:3]  # if RGBA, take only RGB

    if args.apply_resize == "true":
        antialias = (args.antialias == "true")

        if args.crop > 0:
            # pre-crop the image to reduce the processing time
            img = center_crop(img, int(args.crop/args.r) + 5)
            
        img = resize(img, z=args.r, interp=args.interp, antialias=antialias)


    # pre-crop the image to ensure the image is squared
    crop_size = min(img.shape[0], img.shape[1])
    img = img[:crop_size, :crop_size]
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
    elif (args.input.lower().endswith(".jpg") or args.input.lower().endswith(".jpeg")) and args.apply_resize == "false":
        suppress_jpeg = True
    elif args.suppress_jpeg == "true":
        suppress_jpeg = True
    else:
        suppress_jpeg = False


    # if args.apply_recompress == "true":
    #     img = jpeg2luminance("img_orig.jpg")
    # else:
    #     img = rgb2luminance(img)

    img = img[:,:,1]  # take Green channel
    # img = img[:,:,2]  # debug: take Red channel


    # direction = "horizontal"
    # if args.direction == "h":
    #     N = img.shape[1]
    #     direction = "horizontal"
    #     # max_period = img.shape[1]
    # if args.direction == "v":
    #     N = img.shape[0]
    #     direction = "vertical"
    #     # max_period = img.shape[0]

    H, W = img.shape[0], img.shape[1]
    N = W  # take the width as the principal axis

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
    nfa_hori, img_preproc = detect_resampling(
            img, preproc=preproc, preproc_param=preproc_param, window_ratio=window_ratio, 
            nb_neighbor=nb_neighbor, direction="horizontal", suppress_jpeg=suppress_jpeg, max_period=W, return_preproc=True)

    nfa_verti, img_preproc = detect_resampling(
            img_preproc, preproc="none", preproc_param=preproc_param, window_ratio=window_ratio, 
            nb_neighbor=nb_neighbor, direction="vertical", suppress_jpeg=suppress_jpeg, max_period=H, return_preproc=True)

    val_range = 3
    nfa = cross_validate(nfa_hori, nfa_verti, val_range)

    # debug
    # nfa = nfa_hori

    # save preprocessed image
    image.imsave("img_preproc.png", img_preproc, cmap='gray')

    # save spec
    spec = np.fft.fft2(img_preproc, norm="ortho", axes=(0,1))
    image.imsave("spec_preproc.png", np.log(np.abs(spec) + 1))

    lognfa = np.log10(nfa + 1e-90)
    log_threshold = LOG_THRESHOLDS[args.preproc]

    if suppress_jpeg:
        lognfa = filter_by_nms(lognfa, log_threshold, np.round(np.arange(1,8)*N/8).astype(int), max_period=N)

    min_lognfa = np.min(lognfa)
    if min_lognfa - log_threshold >= 0:
        prob_cat = 0
        prob_text = "unlikely"
    elif min_lognfa - log_threshold >= -2:
        prob_cat = 1
        prob_text = "possible"
    elif min_lognfa - log_threshold < -2:
        prob_cat = 2
        prob_text = "very likely"

    
    with open("llk_resampled.txt", 'w') as f:
        f.write(f"{prob_cat:d}\n")
        f.write("\n")
        f.write("(The number indicates the likelihood that the image has been resampled: 0 for unlikely, 1 for possible, 2 for very likely)\n")

    orig_sz_arr = np.arange(0, len(lognfa))

    pairs_d_nfa = []

    fig = plt.figure()
    plt.plot(orig_sz_arr, lognfa, "-", label="nfa")
    for i in range(len(orig_sz_arr)):
        d = orig_sz_arr[i]
        if lognfa[i] < log_threshold:
            pairs_d_nfa.append((i, float(nfa[i])))
            if lognfa[i+1] < log_threshold:
                plt.text(d - 18, lognfa[i], f"{d}", c='r')
            else:
                plt.text(d + 3, lognfa[i], f"{d}", c='r')
            plt.scatter(d, lognfa[i], s=10, color="r")
    plt.ylabel(r'$log_{10} \, NFA(d)$', fontsize=12)
    plt.xlabel("tested spectral correlation distance " + r'$d$', fontsize=12)

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
    ranges_sorted, logits_sorted = predict_rate_range(img, config=Config())  # already sorted by logits
    # print("logits:", logits)
    # print("ranges:", ranges)
    if logits_sorted[0] > 0 and logits_sorted[1] < 0: # only one possible range
        ranges_jpeg = [ranges_sorted[0]]
    else:
        ranges_jpeg = [ranges_sorted[0], ranges_sorted[1]]

    ranges_non_jpeg = [ranges_sorted[0], ranges_sorted[1], ranges_sorted[2]]

    d_list = [d for d,_ in pairs_d_nfa]

    M_list = []

    if len(d_list) > 2:  # the image was likely to be a JPEG image
        # print("The original image is likely to be a JPEG image")
        for rate_range in ranges_jpeg:
            # M_list.extend(estimate_original_size_jpeg(d_list, N, rate_range, eps=2))
            M_list.extend(estimate_original_size_by_jpegx16(d_list, N, rate_range, eps=2))
        if verify_upsample_by_2_pow_n(d_list, N, n=5):
            M_list.append(N / 32)
        elif verify_upsample_by_2_pow_n(d_list, N, n=4):
            M_list.append(N / 16)
    else:
        # print("The original image is unlikely to be a JPEG image")
        for rate_range in ranges_non_jpeg:
            M_list.extend(estimate_original_size_non_jpeg(d_list[:2], N, rate_range, eps=2))
        
    if len(d_list) > 0 and  all(abs(d - N/2) <= 2 for d in d_list):
        only_demosaic = True
    else:
        only_demosaic = False

    print("The method is run successfully in {:.2f} s.".format(time.time() - start))
    print()
    print("It is {} that tested image has been resampled.".format(prob_text))
    print()
    print_results(N, M_list, only_demosaic)


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


def print_results(N:int, M_candidates:list, only_demosaic=False):
    list_N_M_r = []
    if len(M_candidates) == 0:
        print("WANING: the following cases are possible:")
        print("- The image is an original image without being resampled;") 
        print("- The image has been severly downsampled so that the resampling traces are undetectable;")
        print("- The image has been resampled but severely post-compressed so that the resampling traces are undetectable.")
    else:
        # print("Image resampling traces are detected.")
        # print()
        if only_demosaic:
            print("If this is an uncompressed image, the detected resampling traces seem to be left by demosaicing. No other resampling traces are detected.")
            print()
        else:
            print(f"The tested image in N x N = {N} x {N} has been resampled from an original image in M x M, with M being one of the following values:")
            for M in M_candidates:
                r = N / M
                if type(M) is int:
                    print(f"  M={M:d}, resampling rate r=N/M={r:.2f}")
                elif type(M) is float:
                    print(f"  M={M:.1f}, resampling rate r=N/M={r:.2f}")
                list_N_M_r.append((N, M, r))
    write_tuples_to_txt(list_N_M_r, ["current_size", "estimated_original_size", "estimated_resample_rate"], "estimated_rates.txt")


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
    # main(args)
    main_bidirection(args)  # debug

    ## RUN COMMAND
    # python main_ipol.py input_0.png -p $p --apply_resize $apply_resize --r $r --interp $interp 
    # e.g. (local) python main_ipol.py ../test/plot_result/baboon.png -p $p --apply_resize $apply_resize -r $r --interp $interp --antialias $antialias
