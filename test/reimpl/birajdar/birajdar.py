""" 
Reimplementation of the detection method described in "Blind Authentication of Resampled Images and Rescaling Factor Estimation" by Gajanan K. Birajdar and Vijay H. Mankar.
"""

import glob
import os
import sys
from tqdm import tqdm

import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import pandas as pd


def zero_cross(img:np.ndarray):
    p = np.zeros_like(img).astype(float)
    cross_map = (img[:,:-1] * img[:,1:] <= 0)
    p[:,:-1][cross_map] = 1
    return p
    

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


def Birajdar(img:np.ndarray, num_avg:int, is_jpeg=False):
    assert len(img.shape) == 2

    img = img.astype(float)
    
    # Sn = np.zeros_like(img)
    
    Sn = derivative(img, order=2)

    # print("Sn", Sn)

    S_zc = zero_cross(Sn)
    S_zc = np.mean(S_zc, axis=0)

    S_zc_fft = np.abs(np.fft.fft(S_zc, norm="ortho"))

    # print(S_zc_fft)

    # moving average
    # num_avg = 15
    half_num_avg = (num_avg - 1) // 2
    
    # local maxima mask
    argmax = argrelmax(S_zc_fft)[0]
    mask_maxima = np.zeros(len(S_zc_fft), dtype=bool)
    mask_maxima[argmax] = True

    # thresdhold
    local_avg = local_average(S_zc_fft, nb_avg=num_avg)
    
    # plt.plot(S_zc_fft)
    # plt.plot(local_avg, "--")
    # plt.show()

    saliency_map = S_zc_fft / (local_avg + 1e-10)  # proportion
    saliency_map[:half_num_avg] = 0
    saliency_map[-half_num_avg:] = 0
    saliency_map[~mask_maxima] = 0

    if is_jpeg:
        for i in range(1, 8):
            saliency_map[ int(i / 8 * len(saliency_map)) ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) + 1 ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) - 1 ] = 0

    return saliency_map


# At FPR = 1%, threshold = 3.06
def estimate_period(img, num_avg, threshold) -> int:
    saliency_map = Birajdar(img, num_avg=num_avg)
    saliency_map[:num_avg//2] = 0
    saliency_map[-num_avg//2:] = 0
    period_most_likely = np.argmax(saliency_map)
    if saliency_map[period_most_likely] > threshold:
        return period_most_likely
    else:
        return -1


def test_Birajdar():
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r001d260dt.png"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/reimpl/flower_1.1.png"
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r0225b88bt.png"
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_-1/r0225b88bt.png"
    # fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r0f91c753t.png"
    fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_1.50/bicubic/jpeg_q1-1/jpeg_q295/r00617aa1t.jpg"
    img = skimage.io.imread(fname)
    num_avg = 15
    saliency_map = Birajdar(img, num_avg, is_jpeg=True)
    
    # plt.plot(fft)
    # plt.show()

    # plt.plot(saliency_map[num_avg//2: -num_avg//2])
    plt.plot(saliency_map)
    plt.show()

    # print(fft)


def dilate(x:np.ndarray, rg:int):
    x_ero = x.copy()
    for r in range(1, rg+1):
        x_ero[:-r] = np.maximum(x_ero[:-r], x[r:])
        x_ero[r:] = np.maximum(x_ero[r:], x[:-r])

    return x_ero


def cross_validate(sal_0:np.ndarray, sal_1:np.ndarray, rg:int):
    """ Make agreement between two saliency map, a significant detection from one array must be validated by another significant detection from the other array that is close to the first detection.

    Parameters
    ----------
    sal_0 : np.ndarray
        the first saliency array
    sal_1 : np.ndarray
        the second saliency array
    rg : int
        the range of the cross validation, such that out(x) = min_r( max( sal_0(x), sal_1(x+r) ) )
    """
    assert len(sal_0.shape) == 1 and len(sal_1.shape) == 1 and sal_0.shape == sal_1.shape

    # erode nfa_1
    sal_1_dil = dilate(sal_1, rg)

    return np.maximum(sal_0, sal_1_dil)



def main_experiment_classification(ratio:float, bidirect:bool):
    num_neighbor = 15

    target_size = 600
    # ratio = 1.3
    interp = "bicubic"
    jpeg_q1 = -1
    jpeg_q2 = 95

    input_folder = f"/Volumes/HPP900/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"

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
            is_jpeg = False
        else:
            is_jpeg = True
        
        if bidirect:
            saliency_map_0 = Birajdar(img, num_neighbor, is_jpeg=is_jpeg)
            saliency_map_1 = Birajdar(img.T, num_neighbor, is_jpeg=is_jpeg)
            saliency_map = cross_validate(saliency_map_0, saliency_map_1, rg=3)
        else:
            saliency_map = Birajdar(img, num_neighbor, is_jpeg=is_jpeg)

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

    if bidirect:
        output_fname = f"birajdar_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_bidirect_resa_ratio_{ratio:.2f}.csv"
    else:
        output_fname = f"birajdar_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"
    df.to_csv(output_fname, index=False, float_format='%4.2f')


def main_experiment_estimation(ratio:float):
    num_neighbor = 15

    target_size = 600
    # ratio = 1.3
    interp = "bicubic"
    jpeg_q1 = -1
    jpeg_q2 = 90

    threshold = 3.06  # threshold at 1% FPR

    input_folder = f"/Volumes/HPP900/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    fname_list = glob.glob(os.path.join(input_folder, "*.png"))
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
    df_data["estimated_period"] = []
    # df_data["period"] = []


    for fname in tqdm(fname_list):
        img = skimage.io.imread(fname)

        estimated_period = estimate_period(img, num_neighbor, threshold=threshold)

        df_data["fname"].append(fname)
        df_data["target_size"].append(target_size)
        df_data["resa_ratio"].append(ratio)
        df_data["interp"].append(interp)
        df_data["jpeg_q1"].append(-1)
        df_data["jpeg_q2"].append(-1)
        df_data["num_neighbor"].append(num_neighbor)
        df_data["estimated_period"].append(estimated_period)

    # save to .csv
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,num_neighbor,saliency
    df = pd.DataFrame(df_data)
    print(df)

    np.set_printoptions(precision=3)

    output_fname = f"birajdar_resa_ratio_{ratio:.2f}_period_estimation.csv"
    df.to_csv(output_fname, index=False, float_format='%4.2f')


if __name__ == "__main__":
    # test_Birajdar()

    # print(sys.argv)
    # ratio = float(sys.argv[1])

    for ratio in np.arange(0.6, 1.6+1e-5, 0.1):
        main_experiment_classification(ratio, bidirect=True)
        # main_experiment_estimation(ratio)
