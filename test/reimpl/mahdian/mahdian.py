""" mahdian.py
    Reimplementation of the method described in "Blind Authentication Using Periodic Properties of Interpolation"
    by Babak Mahdian and Stanislav Saic.
"""

import glob
import os
import sys
from tqdm import tqdm

import numpy as np
import skimage.io
import scipy
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import pandas as pd


# 1. derivative order 2
# 2. radon transform -> phi_theta
#   derivative order 1
#   auto-covariance
#   fft
# 3. search for periodicity

PLOT_HISTO = False
PLOT_FOLDER = "plots"
THRESHOLD = 8.1

def example_radon():
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon, rescale

    image = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    print("sinogram")
    print(sinogram.shape)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(
        sinogram,
        cmap=plt.cm.Greys_r,
        extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
        aspect='auto',
    )

    fig.tight_layout()
    plt.show()


def auto_cov(x:np.ndarray):
    from scipy import signal
    x_normalized = x - np.mean(x)

    return signal.correlate(x_normalized, x_normalized)


def radon_transform(img:np.ndarray):
    return


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


def autocorrelate(x:np.ndarray):
    assert len(x.shape) == 1
    x_fft = np.fft.fft(x)
    x_autocorr = np.fft.ifft( x_fft * np.conjugate(x_fft) )
    return x_autocorr


def test_autocorrelate():
    x = np.array([0, 1, 2, 3])
    x_autocorr = autocorrelate(x)
    print(x)
    print(x_autocorr)


def mahdian_detector(img:np.ndarray, num_avg:int, is_jpeg:bool):
    assert len(img.shape) == 2
    img = img.astype(float)
    Dn = derivative(img, order=2)

    phi = np.mean(np.abs(Dn), axis=0)
    # phi = np.var(Dn, axis=0)

    phi = phi - np.mean(phi)
    # plt.plot(phi)
    # plt.title("phi")
    # plt.show()

    # # debug: check this version
    # phi_fft = np.fft.fft(phi, norm="ortho")
    # Rphi_fft = phi_fft * np.conj(phi_fft)
    # Rphi_fft = (phi_fft * np.conj(phi_fft)).real
    # # Rphi_fft = np.abs(phi_fft) #  * np.conj(phi_fft)

    # original version
    Rphi = autocorrelate(phi).real
    Rphi_fft = np.abs(np.fft.fft(Rphi, norm="ortho"))


    # plt.plot(Rphi_fft)
    # plt.title("R_phi")
    # plt.show()

    
    # # debug: check the fft
    # plt.figure()
    # plt.plot(Rphi_fft)
    # plt.title("|FFT(R_phi)|")
    # plt.show()

    # moving average
    half_num_avg = (num_avg - 1) // 2
    
    # local maxima mask
    argmax = argrelmax(Rphi_fft)[0]
    mask_maxima = np.zeros(len(Rphi_fft), dtype=bool)
    mask_maxima[argmax] = True

    # thresdhold
    local_avg = local_average(Rphi_fft, nb_avg=num_avg)

    # plt.plot(vp_f)
    # plt.plot(local_avg, "--")
    # plt.show()

    saliency_map = Rphi_fft / (local_avg + 1e-10)  # proportion
    saliency_map[:half_num_avg] = 0
    saliency_map[-half_num_avg:] = 0
    
    # debug: deactivate this to see the difference
    saliency_map[~mask_maxima] = 0

    # debug: deactivate to see the jpeg saliency map
    if is_jpeg:
        for i in range(1, 8):
            saliency_map[ int(i / 8 * len(saliency_map)) ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) + 1 ] = 0
            saliency_map[ int(i / 8 * len(saliency_map)) - 1 ] = 0

    return saliency_map


def dilate(x:np.ndarray, rg:int):
    x_dil = x.copy()
    for r in range(1, rg+1):
        x_dil[:-r] = np.maximum(x_dil[:-r], x[r:])
        x_dil[r:] = np.maximum(x_dil[r:], x[:-r])

    return x_dil


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

    # dilate
    sal_1_dil = dilate(sal_1, rg)

    return np.maximum(sal_0, sal_1_dil)


def main_experiment(ratio:float, bidirect:bool, antialias:int):
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
    print("input_folder:", input_folder)

    if jpeg_q2 == -1:
        is_jpeg = False
        fname_list = glob.glob(os.path.join(input_folder, "*.png"))
    else:
        is_jpeg = True
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

    if PLOT_HISTO:
        plot_sub_folder = os.path.join(PLOT_FOLDER, f"r_{ratio:.2f}")
        os.makedirs(plot_sub_folder, exist_ok=True)

    for fname in tqdm(fname_list):
        img = skimage.io.imread(fname)

        if bidirect:
            saliency_map_0 = mahdian_detector(img, num_neighbor, is_jpeg)
            saliency_map_1 = mahdian_detector(img.T, num_neighbor, is_jpeg)
            saliency_map = cross_validate(saliency_map_0, saliency_map_1, rg=3)
        else:
            saliency_map = mahdian_detector(img, num_neighbor, is_jpeg)


        saliency = np.max(saliency_map[num_neighbor//2: -num_neighbor//2])

        df_data["fname"].append(fname)
        df_data["target_size"].append(target_size)
        df_data["resa_ratio"].append(ratio)
        df_data["interp"].append(interp)
        df_data["jpeg_q1"].append(jpeg_q1)
        df_data["jpeg_q2"].append(jpeg_q2)
        df_data["num_neighbor"].append(num_neighbor)
        df_data["saliency"].append(saliency)


        if PLOT_HISTO:
            print("saliency_map:", saliency_map.shape)
            plt.figure(figsize=(10,7))
            plt.plot(saliency_map, ".-")
            for i in range(len(saliency_map)):
                # Plot each point with color depending on the threshold
                if saliency_map[i] > THRESHOLD:
                    plt.scatter(i, saliency_map[i], color='red')
                    plt.text(i + 2, saliency_map[i], f"{i}", color='red', fontsize=8)
            plt.axhline(y=THRESHOLD, color='red', linestyle='--')

            plt.xlabel("frequency")
            plt.ylabel("saliency")
            plt.title(fname, fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_FOLDER, f"r_{ratio:.2f}", os.path.basename(fname)[:-4] + ".png"))
            # plt.show()
            # exit(0)

    # save to .csv
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,num_neighbor,saliency
    df = pd.DataFrame(df_data)
    print(df)

    np.set_printoptions(precision=3)

    if antialias == 0:
        output_fname = f"mahdian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"
    elif antialias == 1:
        output_fname = f"mahdian_antialias_gaussian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"
    elif antialias == 2:
        output_fname = f"mahdian_antialias_matlab_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"

    # if bidirect:
    #     output_fname = f"mahdian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_bidirect_resa_ratio_{ratio:.2f}.csv"
    # else:
    #     output_fname = f"mahdian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_resa_ratio_{ratio:.2f}.csv"

    df.to_csv(output_fname, index=False, float_format='%4.2f')
    print(f"Saved results at {output_fname}")


def test_mahdian_detector():
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r0225b88bt.png"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_0.90/bicubic/jpeg_-1/r0f91c753t.png"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_-1/r00a940e1t.png"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r00a940e1t.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r000da54ft.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r00617aa1t.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r01d9da09t.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r001d260dt.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q290/r0b61ed37t.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r1d79b6e7t.jpg"
    # fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q2-1/r00617aa1t.png"
    fname = "/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/noise_target_size_600/r_1.20/bicubic/jpeg_q1-1/jpeg_q2-1/001.png"

    img = skimage.io.imread(fname)
    nb_avg = 15
    saliency_map = mahdian_detector(img, num_avg=nb_avg, is_jpeg=True)

    plt.title("saliency map")
    # plt.plot(saliency_map[nb_avg//2: -nb_avg//2])
    plt.plot(saliency_map)
    plt.show()


if __name__ == "__main__":
    # example_radon()
    # test_autocorrelate()
    # test_mahdian_detector()

    # print(sys.argv)
    # ratio = float(sys.argv[1])

    for ratio in np.arange(0.6, 1.0+1e-5, 0.05):
    # for ratio in [1.40]:
        main_experiment(ratio, bidirect=False, antialias=1)
