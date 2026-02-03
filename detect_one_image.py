import time
import os
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt
import iio

from src.ird import detect_resampling, detect_resampling_with_cross_val
from src.misc import rgb2luminance


WINDOW_RATIO = 0.10
NB_NEIGHBOR = 20

LOG_THRESHOLDS = {
    "rt": -5,
    "tv": -7,
    "dct": -19,
    "phot": -12,
    "none": -17,
}


def preprocess_image(args):
    img = iio.read(args.input)
    img = rgb2luminance(img)

    if args.crop is not None:
        x, y, w, h = tuple(args.crop)
        img = img[y:y + h, x:x + w]

    return img


def detect_resampling_wrapper(img, preproc, preproc_param, direction, is_jpeg):
    if direction in ["horizontal", "vertical"]:
        if direction == "horizontal":
            max_distance = img.shape[1] - 1
        else:
            max_distance = img.shape[0] - 1
        nfa, preprocessed_img = detect_resampling(
            img, preproc=preproc, preproc_param=preproc_param, window_ratio=WINDOW_RATIO,
            nb_neighbor=NB_NEIGHBOR, direction=direction,
            is_jpeg=is_jpeg, is_demosaic=False,
            max_distance=max_distance, 
            return_preproc_img=True
        )       
    else:  # both
        nfa, preprocessed_img = detect_resampling_with_cross_val(
            img, preproc=preproc, preproc_param=preproc_param, window_ratio=WINDOW_RATIO,
            nb_neighbor=NB_NEIGHBOR,
            is_jpeg=is_jpeg, is_demosaic=False,
            return_preproc_img=True
        )
    
    return nfa, preprocessed_img


def get_preprocessing_params(args):
    if args.preproc == "rt":
        return "rt", {"rt_size": 3}
    elif args.preproc == "tv":
        return "tv", None
    elif args.preproc == "dct":
        return "dct", {"sigma": 3}
    elif args.preproc == "phot":
        return "phot", None
    elif args.preproc == "none":
        return "none", None


def visualize_results(nfa, preproc, save_path=None):
    lognfa = np.log10(nfa + 1e-50)
    orig_sz_arr = np.arange(0, len(lognfa))

    fig = plt.figure(figsize=(5, 3.1))
    # fig = plt.figure(figsize=(8, 2.5))
    plt.plot(orig_sz_arr, lognfa, "-", label="nfa")

    n = 16
    idx_list = np.argsort(lognfa)[:n]
    idx_list = np.sort(idx_list)

    threshold = LOG_THRESHOLDS[preproc]
    for idx in idx_list:
        d = orig_sz_arr[idx]
        if lognfa[idx] < threshold:
            offset = -18 if lognfa[idx + 1] < threshold else 3
            plt.text(d + offset, lognfa[idx], f"{d}", c='r')
            plt.scatter(d, lognfa[idx], s=10, color="r")

    if np.min(lognfa) > -3:
        plt.ylim(bottom=-3)
    plt.ylabel(r'$log_{10} \, NFA(d)$', fontsize=12, labelpad=0)
    plt.xlabel(r'$d$', fontsize=12)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Save to file
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def main(args):
    print("args:", args)

    # Load and preprocess the image
    img = preprocess_image(args)

    # Determine direction and max period
    if args.direction == "h":
        direction = "horizontal"
    elif args.direction == "v":
        direction = "vertical"
    elif args.direction == "both":
        direction = "both"

    # Set preprocessing parameters
    preproc, preproc_param = get_preprocessing_params(args)

    is_jpeg = args.is_jpeg

    start = time.time()

    # Detect resampling
    nfa, preprocessed_img = detect_resampling_wrapper(img, preproc, preproc_param, direction, is_jpeg)

    print("Detection finished. Spent time: {:.3f} seconds".format(time.time() - start))

    # Calculate values
    threshold = LOG_THRESHOLDS[preproc]
    log_values = np.log10(nfa + 1e-50)
    anomalous_distances = np.where(log_values < threshold)[0]

    # Header and Table structure
    print(f"\n═══ DETECTION REPORT ═══")
    print(f"Threshold: {threshold}")
    print(f"Anomalous distances found: {len(anomalous_distances)}")
    print()

    # Table Top
    print("┌──────────────┬────────────────────────┐")
    print("│   Distance   │       log10(NFA)       │")
    print("├──────────────┼────────────────────────┤")

    if len(anomalous_distances) == 0:
        print("│            ( No Anomalies )           │")
    else:
        for d in anomalous_distances:
            val = log_values[d]
            # Using f-string padding for perfect alignment
            print(f"│ {d:^12} │ {val:^22.4f} │")

    # Table Bottom
    print("└──────────────┴────────────────────────┘")
    print()

    # Create output folder if it doesn't exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Create a subfolder named after the current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.out_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    # Save the preprocessed image as a grayscale PNG
    preprocessed_img_path = os.path.join(save_folder, f"{os.path.basename(args.input)}_{preproc}_preprocessed.png")
    iio.write(preprocessed_img_path, ((preprocessed_img - preprocessed_img.min()) / (preprocessed_img.max() - preprocessed_img.min()) * 255).astype(np.uint8))
    print(f"Saved preprocessed image to: {preprocessed_img_path}")

    save_path = os.path.join(save_folder, f"{os.path.basename(args.input)}_{preproc}_nfa.pdf")

    # Visualize results
    visualize_results(nfa, preproc, save_path=save_path)

    # Save parameters to hparam.yaml
    hparam_path = os.path.join(save_folder, "hparam.yaml")
    with open(hparam_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"Saved parameters to: {hparam_path}")

    # Save distance and NFA values to nfa.txt, each row: distance, nfa
    nfa_txt_path = os.path.join(save_folder, "nfa.txt")
    with open(nfa_txt_path, "w") as f:
        f.write(f"distance, nfa\n")
        for distance, nfa_value in enumerate(nfa):
            f.write(f"{distance}, {nfa_value}\n")
    print(f"Saved NFA values to: {nfa_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Resampling Detector")
    parser.add_argument("input", type=str, help="Path of the input file.")
    parser.add_argument("--direction", type=str, default="h", choices=["h", "v", "both"],
                        help="Direction for resampling detection.")
    parser.add_argument("--preproc", type=str, default="rt", 
                        choices=["rt", "tv", "dct", "phot", "none"],
                        help="Preprocessing method.")
    parser.add_argument("--is_jpeg", action="store_true", default=False,
                        help="Whether the input image is a JPEG compressed image.")
    parser.add_argument('--crop', nargs=4, type=int, metavar=("x", "y", "w", "h"), default=None, 
                        help="Crop the image to the rectangle defined by (x, y, w, h).")
    parser.add_argument("--out_folder", type=str, default="results", 
                        help="Folder to save results.") 
    args = parser.parse_args()

    main(args)


# Usage:
#   python detect_one_image.py img/baboon.png --direction h --preproc rt
#   python detect_one_image.py img/baboon_1.3.png --direction h --preproc rt