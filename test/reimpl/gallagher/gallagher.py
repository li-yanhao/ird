""" 
Reimplementation of the detection method described in "Detection of Linear and Cubic Interpolation in JPEG Compressed Images" by Andrew C. Gallagher.
"""

import glob
import os
import sys
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.signal import argrelmax
import pandas as pd


def derivative(img:np.ndarray, order):
    if order == 0:
        return img
    else:
        img_1 = derivative(img, order-1)
        Sn = np.zeros((img_1.shape[0], img_1.shape[1]-1))
        Sn = img_1[:, 1:] - img_1[:, :-1]
        return Sn


def local_average(x, nb_avg:int):
    assert nb_avg >= 1 and nb_avg % 2 == 1
    half = (nb_avg - 1) // 2
    x_cum = np.cumsum(x)
    
    y = np.zeros(len(x), dtype=float)
    # y = x_cum[nb_avg:] - x_cum[:-nb_avg]

    y[half] = x_cum[nb_avg]
    y[half+1 : -half] = x_cum[nb_avg:] - x_cum[:-nb_avg]

    return y / nb_avg

    # cum: i-th: sum until i
    # i-th: i-half ~ i + half


def Gallagher(img:np.ndarray, num_avg:int, is_jpeg:bool):
    assert len(img.shape) == 2
    img = img.astype(float)
    sp = derivative(img, order=2)
    vp = np.mean(np.abs(sp), axis=0)
    vp_f = np.abs(np.fft.fft(vp, norm="ortho"))

    # moving average
    # num_avg = 15
    half_num_avg = (num_avg - 1) // 2
    
    # local maxima mask
    argmax = argrelmax(vp_f)[0]
    mask_maxima = np.zeros(len(vp_f), dtype=bool)
    mask_maxima[argmax] = True
    # print(mask_maxima[54])

    # thresdhold
    local_avg = local_average(vp_f, nb_avg=num_avg)

    # plt.plot(vp_f)
    # plt.plot(local_avg, "--")
    # plt.show()

    saliency_map = vp_f / (local_avg + 1e-10)  # proportion
    # print(saliency_map)
    saliency_map[:half_num_avg] = 0
    saliency_map[-half_num_avg:] = 0
    saliency_map[~mask_maxima] = 0

    # print(saliency_map[54])

    if is_jpeg:
        for i in range(1, 8):
            saliency_map[ int(i / 8 * len(saliency_map)) ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) + 1 ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) - 1 ] = 0

    return vp_f, saliency_map


def test_Gallagher():
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r001d260dt.png"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/reimpl/flower_1.1.png"
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.70/bicubic/jpeg_-1/r0225b88bt.png"
    fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_-1/r00a940e1t.png"
    
    img = skimage.io.imread(fname)
    nb_avg = 15
    fft, propor_map = Gallagher(img, num_avg=nb_avg)

    # plt.plot(fft)
    # plt.show()

    plt.plot(propor_map)
    plt.show()

    # print(fft)
    print(propor_map)



def main_experiment(ratio:float, antialias:int):
    num_neighbor = 15

    target_size = 600
    interp = "bicubic"
    jpeg_q1 = -1
    jpeg_q2 = -1

    if antialias == 0:
        input_folder = f"/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    elif antialias == 1:
        input_folder = f"/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    elif antialias == 2:
        input_folder = f"/Users/yli/phd/synthetic_image_detection/ird/test/data/resample_detection/matlab/target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"

    if jpeg_q2 == -1:
        fname_list = glob.glob(os.path.join(input_folder, "*.png"))
    else:
        fname_list = glob.glob(os.path.join(input_folder, "*.jpg"))

    fname_list.sort()

    
    df_data = {}
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,num_neighbor,saliency
    df_data["fname"] = []
    df_data["target_size"] = []
    df_data["resa_ratio"] = []
    df_data["interp"] = []
    df_data["jpeg_q1"] = []
    df_data["jpeg_q2"] = []
    df_data["num_neighbor"] = []
    df_data["saliency"] = []
    # df_data["period"] = []


    for fname in tqdm(fname_list):
        img = skimage.io.imread(fname)

        if jpeg_q2 == -1:
            fft, saliency_map = Gallagher(img, num_neighbor, is_jpeg=False)
        else:
            fft, saliency_map = Gallagher(img, num_neighbor, is_jpeg=True)

        saliency = np.max(saliency_map[num_neighbor//2: -num_neighbor//2])

        df_data["fname"].append(fname)
        df_data["target_size"].append(target_size)
        df_data["resa_ratio"].append(ratio)
        df_data["interp"].append(interp)
        df_data["jpeg_q1"].append(-1)
        df_data["jpeg_q2"].append(-1)
        df_data["num_neighbor"].append(num_neighbor)
        df_data["saliency"].append(saliency)

    # save to .csv
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,num_neighbor,saliency
    df = pd.DataFrame(df_data)
    print(df)

    np.set_printoptions(precision=3)

    if antialias == 0:
        output_fname = f"gallagher_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"
    elif antialias == 1:
        output_fname = f"gallagher_antialias_gaussian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"
    elif antialias == 2:
        output_fname = f"gallagher_antialias_matlab_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"

    df.to_csv(output_fname, index=False, float_format='%4.2f')


if __name__ == "__main__":
    # test_Gallagher()
    
    # print(sys.argv)
    # ratio = float(sys.argv[1])

    for ratio in np.arange(0.6, 1.0+1e-5, 0.05):
        main_experiment(ratio, antialias=1)
