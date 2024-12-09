import glob
import os
from tqdm import tqdm
import sys

import cv2
import numpy as np
import skimage.io
import pandas as pd


class ResamplingPopescu(object):
    # tool layout
    def __init__(self):
        # self.imagegray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # self.imagegray = self.imagegray - self.imagegray.min()
        # self.imagegray = self.imagegray / self.imagegray.max()

        # self.imagegray = image

        # self.imagegray_nomalized_copy = self.imagegray.copy()
        # self.imagegray_copy_for_probabilitymaps = self.imagegray.copy()
        # self.userfourplot = self.imagegray.copy()

        self.selected_points_probability = []
        self.selected_points_fourier = []
        self.probability_maps = []
        self.fourier_maps = []
        # self.original_sizes_widgets = {}

        self.filter_3x3_Check = True
        self.filter_5x5_Check = False

        self.hanning_check = False
        self.rotationally_invariant_window_check = True
        self.upsample_check = False
        self.center_four_check = False
        self.simple_highpass_check = False
        self.complex_highpass_check = True
        
        self.gamma_spin = 4
        self.rescale_check = True

    def calculate_probability_map(self, image):
        # to do, if probability maps already calculated, do not calculate again;
        # untill then, make fourier maps empty before calculating again:
        # self.imagegray_copy_for_probabilitymaps = self.imagegray_nomalized_copy.copy()

        image = image - image.min()
        image = image / image.max()
        
        if self.filter_5x5_Check:
            p_map = self.calculate_probability_map_5x5(image)
        else:
            p_map = self.calculate_probability_map_3x3(image)

        # self.probability_maps.append(processed_part)
        # self.imagegray_copy_for_probabilitymaps = processed_part
        
        
        return p_map

    def calculate_probability_map_3x3(self, process_part):
        a = np.random.rand(8)
        a = a / a.sum()
        s = 0.005
        d = 0.1
        F, f = self.build_matrices_for_processing_3x3(process_part)
        c = 0
        w = np.zeros(F.shape[0])

        while c < 100:
            s2 = 0
            k = 0

            for y in range(1, process_part.shape[0] - 1):
                for x in range(1, process_part.shape[1] - 1):
                    r = self.compute_residual_3x3(a, process_part, x, y)
                    g = np.exp(-(r ** 2) / s)
                    w[k] = g / (g + d)
                    s2 = s2 + w[k] * r ** 2
                    k = k + 1

            s = s2 / w.sum()
            a2 = np.linalg.inv(F.T * w * w @ F + np.eye(F.shape[1])*1e-10) @ F.T * w * w @ f

            if np.linalg.norm(a - a2) < 0.01:
                break
            else:
                a = a2
                c = c + 1

        return w.reshape(process_part.shape[0] - 2, process_part.shape[1] - 2)

    def calculate_probability_map_5x5(self, process_part):
        a = np.random.rand(24)
        a = a / a.sum()
        s = 0.005
        d = 0.1
        F, f = self.build_matrices_for_processing_5x5(process_part)
        c = 0
        w = np.zeros(F.shape[0])

        while c < 100:
            s2 = 0
            k = 0

            for y in range(2, process_part.shape[0] - 2):
                for x in range(2, process_part.shape[1] - 2):
                    r = self.compute_residual_5x5(a, process_part, x, y)
                    g = np.exp(-(r ** 2) / s)
                    w[k] = g / (g + d)
                    s2 = s2 + w[k] * r ** 2
                    k = k + 1

            s = s2 / w.sum()
            a2 = np.linalg.inv(F.T * w * w @ F) @ F.T * w * w @ f

            if np.linalg.norm(a - a2) < 0.01:
                break
            else:
                a = a2
                c = c + 1

        return w.reshape(process_part.shape[0] - 4, process_part.shape[1] - 4)

    def calculate_fourier_maps(self):
        # to do, if fouriermap already calculated, do not calculate again
        # untill then, make fourier maps empty before calculating again:
        self.fourier_maps = []
        self.calculate_fourier_button.setEnabled(False)  # wait for processing

        num_plots = int(len(self.selected_points_fourier) / 2) + len(
            self.probability_maps
        )
        self.canvas_fourier_maps.figure.clf()
        if num_plots > 1:
            self.axes_fourier_maps = self.canvas_fourier_maps.figure.subplots(
                num_plots, 2
            )
        else:
            self.axes_fourier_maps = np.array(
                [
                    [
                        self.canvas_fourier_maps.figure.add_subplot(1, 2, 1),
                        self.canvas_fourier_maps.figure.add_subplot(1, 2, 2),
                    ]
                ]
            )

        i = 0
        counter_selected_prob_points = 0
        for prob_map in self.probability_maps:
            if (
                len(self.probability_maps) == 1
                and len(self.selected_points_probability) == 0
            ):
                self.fourier_maps.append(self.calculate_fourier_map(prob_map))
                self.axes_fourier_maps[i, 0].imshow(
                    prob_map, cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 0].axis("off")
                self.axes_fourier_maps[i, 1].imshow(
                    self.fourier_maps[i], cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 1].axis("off")
                i = i + 1
            else:
                (x1, y1) = self.selected_points_probability[
                    counter_selected_prob_points
                ]
                (x2, y2) = self.selected_points_probability[
                    counter_selected_prob_points + 1
                ]
                self.fourier_maps.append(self.calculate_fourier_map(prob_map))
                self.userfourplot = self.imagegray_copy_for_probabilitymaps.copy()
                self.userfourplot[
                    min(y1, y2) : max(y1, y2) + 1,
                    [
                        x1,
                        (x1 + 1),
                        (x1 - 1),
                        (x1 - 2),
                        (x1 + 2),
                        x2,
                        (x2 + 1),
                        (x2 - 1),
                        (x2 - 2),
                        (x2 + 2),
                    ],
                ] = 0  # y-stripes
                self.userfourplot[
                    [
                        y1,
                        (y1 - 1),
                        (y1 + 1),
                        (y1 - 2),
                        (y1 + 2),
                        y2,
                        (y2 - 1),
                        (y2 + 1),
                        (y2 - 2),
                        (y2 + 2),
                    ],
                    min(x1, x2) : max(x1, x2) + 1,
                ] = 0  # x-stripes
                self.axes_fourier_maps[i, 0].imshow(
                    self.userfourplot, cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 0].axis("off")
                self.axes_fourier_maps[i, 1].imshow(
                    self.fourier_maps[i], cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 1].axis("off")
                i = i + 1
                counter_selected_prob_points = counter_selected_prob_points + 2
        for j in range(0, len(self.selected_points_fourier), 2):
            if j + 1 < len(self.selected_points_fourier):
                (x1, y1) = self.selected_points_fourier[j]
                (x2, y2) = self.selected_points_fourier[j + 1]

                processed_part = self.calculate_fourier_map(
                    self.imagegray_copy_for_probabilitymaps[
                        min(y1, y2) : max(y1, y2) + 1, min(x1, x2) : max(x1, x2) + 1
                    ]
                )
                self.fourier_maps.append(processed_part)
                self.userfourplot = self.imagegray_copy_for_probabilitymaps.copy()
                self.userfourplot[
                    min(y1, y2) : max(y1, y2) + 1,
                    [
                        x1,
                        (x1 + 1),
                        (x1 - 1),
                        (x1 - 2),
                        (x1 + 2),
                        x2,
                        (x2 + 1),
                        (x2 - 1),
                        (x2 - 2),
                        (x2 + 2),
                    ],
                ] = 0  # y-stripes
                self.userfourplot[
                    [
                        y1,
                        (y1 - 1),
                        (y1 + 1),
                        (y1 - 2),
                        (y1 + 2),
                        y2,
                        (y2 - 1),
                        (y2 + 1),
                        (y2 - 2),
                        (y2 + 2),
                    ],
                    min(x1, x2) : max(x1, x2) + 1,
                ] = 0  # x-stripes
                self.axes_fourier_maps[i, 0].imshow(
                    self.userfourplot, cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 0].axis("off")
                self.axes_fourier_maps[i, 1].imshow(
                    self.fourier_maps[i], cmap="gray", vmin=0, vmax=1
                )
                self.axes_fourier_maps[i, 1].axis("off")
                i = i + 1

        # self.figure_four.subplots_adjust(
        #     left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
        # )

        # self.canvas_fourier_maps.figure.canvas.draw()
        # self.calculate_fourier_button.setEnabled(True)

    def make_high_pass_filter(self, shape):
        rows, cols = shape
        H = np.zeros((rows, cols))
        center_x, center_y = rows // 2, cols // 2
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        for i in range(rows):
            for j in range(cols):
                r = (
                    np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    / max_radius
                    * np.sqrt(2)
                )
                if r <= np.sqrt(2):
                    H[i, j] = 0.5 - 0.5 * np.cos(np.pi * r / np.sqrt(2))

        return H

    def make_rotational_invariant_window(self, shape):
        rows, cols = shape
        W = np.zeros((rows, cols))
        center_x, center_y = rows // 2, cols // 2
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        for i in range(rows):
            for j in range(cols):
                r = (
                    np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    / max_radius
                    * np.sqrt(2)
                )
                if r < 3 / 4:
                    W[i, j] = 1
                elif r <= np.sqrt(2):
                    W[i, j] = 0.5 + 0.5 * np.cos(
                        np.pi * (r - 3 / 4) / (np.sqrt(2) - 3 / 4)
                    )

        return W

    def calculate_fourier_map(self, process_prob_map):
        # take center square portion of probability map, this will make analysis easier and more consistent
        # x, y = process_prob_map.shape
        # size = min(x, y)
        # half_size = size // 2
        # center_x, center_y = x // 2, y // 2
        # square_prob_map = process_prob_map[
        #     center_x - half_size : center_x + half_size,
        #     center_y - half_size : center_y + half_size,
        # ]
        square_prob_map = process_prob_map

        # apply pre-processing window
        if self.hanning_check:
            # Apply Hanning window
            hanning_window = np.hanning(square_prob_map.shape[0])[:, None] * np.hanning(
                square_prob_map.shape[1]
            )
            windowed_prob_map = square_prob_map * hanning_window
        elif self.rotationally_invariant_window_check:
            # apply rot. invariant window
            W = self.make_rotational_invariant_window(square_prob_map.shape)
            windowed_prob_map = square_prob_map * W
        else:
            # if none is checked, return
            return

        if self.upsample_check:
            upsampled = cv2.pyrUp(windowed_prob_map)
        else:
            upsampled = windowed_prob_map

        # fourier transform
        dft = np.fft.fft2(upsampled, norm="ortho")
        fourier = np.fft.fftshift(dft)

        # take center?
        if self.center_four_check:
            height, width = fourier.shape
            center_x, center_y = width // 2, height // 2
            half_size = width // 4
            fourier_center = fourier[
                center_y - half_size : center_y + half_size,
                center_x - half_size : center_x + half_size,
            ]
        else:
            fourier_center = fourier

        # filter option
        if self.simple_highpass_check:
            # create circular mask:
            rows, cols = fourier_center.shape
            center = (int(cols / 2), int(rows / 2))
            radius = int(0.1 * (min(rows, cols) / 2))
            if radius == 0:
                radius = 1  # radius should at least be = 1

            Y, X = np.ogrid[:rows, :cols]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

            mask = dist_from_center <= radius
            filtered_spectrum = fourier_center.copy()
            filtered_spectrum[mask] = 0

        elif self.complex_highpass_check:
            H = self.make_high_pass_filter(fourier_center.shape)
            filtered_spectrum = fourier_center * H
        else:
            return

        # scale and gamma correct
        magnitude_spectrum = np.abs(filtered_spectrum)
        scaled_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (
            magnitude_spectrum.max() - magnitude_spectrum.min()
        )

        gamma_corrected_spectrum = np.power(scaled_spectrum, self.gamma_spin)

        # rescale if checked
        if self.rescale_check:
            rescaled_spectrum = gamma_corrected_spectrum * magnitude_spectrum.max()
        else:
            rescaled_spectrum = gamma_corrected_spectrum

        return rescaled_spectrum

    def build_matrices_for_processing_3x3(self, I):
        k = 0
        F = np.zeros(
            ((I.shape[0] - 2) * (I.shape[1] - 2), 8)
        )  # the i^th row of the matrix F corresponds to the pixel neighbors of the i^th element of f
        f = np.zeros(
            (I.shape[0] - 2) * (I.shape[1] - 2)
        )  # the vector f is the image strung out in row-order
        for y in range(1, I.shape[0] - 1):
            for x in range(1, I.shape[1] - 1):
                F[k, 0] = I[y - 1, x - 1]
                F[k, 1] = I[y - 1, x]
                F[k, 2] = I[y - 1, x + 1]
                F[k, 3] = I[y, x - 1]
                # skip I(x; y) corresponding to a(3; 3) which is 0
                F[k, 4] = I[y, x + 1]
                F[k, 5] = I[y + 1, x - 1]
                F[k, 6] = I[y + 1, x]
                F[k, 7] = I[y + 1, x + 1]
                f[k] = I[y, x]
                k = k + 1
        return F, f

    def build_matrices_for_processing_5x5(self, I):
        k = 0
        F = np.zeros(
            ((I.shape[0] - 4) * (I.shape[1] - 4), 24)
        )  # the i^th row of the matrix F corresponds to the pixel neighbors of the i^th element of f
        f = np.zeros(
            (I.shape[0] - 4) * (I.shape[1] - 4)
        )  # the vector f is the image strung out in row-order
        for y in range(2, I.shape[0] - 2):
            for x in range(2, I.shape[1] - 2):
                F[k, 0] = I[y - 2, x - 2]
                F[k, 1] = I[y - 2, x - 1]
                F[k, 2] = I[y - 2, x]
                F[k, 3] = I[y - 2, x + 1]
                F[k, 4] = I[y - 2, x + 2]
                F[k, 5] = I[y - 1, x - 2]
                F[k, 6] = I[y - 1, x - 1]
                F[k, 7] = I[y - 1, x]
                F[k, 8] = I[y - 1, x + 1]
                F[k, 9] = I[y - 1, x + 2]
                F[k, 10] = I[y, x - 2]
                F[k, 11] = I[y, x - 1]
                # skip I(x; y) corresponding to a(3; 3) which is 0
                F[k, 12] = I[y, x + 1]
                F[k, 13] = I[y, x + 2]
                F[k, 14] = I[y + 1, x - 2]
                F[k, 15] = I[y + 1, x - 1]
                F[k, 16] = I[y + 1, x]
                F[k, 17] = I[y + 1, x + 1]
                F[k, 18] = I[y + 1, x + 2]
                F[k, 19] = I[y + 2, x - 2]
                F[k, 20] = I[y + 2, x - 1]
                F[k, 21] = I[y + 2, x]
                F[k, 22] = I[y + 2, x + 1]
                F[k, 23] = I[y + 2, x + 2]
                f[k] = I[y, x]
                k = k + 1
        return F, f

    def compute_residual_3x3(self, a, I, x, y):
        r = I[y, x] - (
            a[0] * I[y - 1, x - 1]
            + a[1] * I[y - 1, x]
            + a[2] * I[y - 1, x + 1]
            + a[3] * I[y, x - 1]
            + a[4] * I[y, x + 1]
            + a[5] * I[y + 1, x - 1]
            + a[6] * I[y + 1, x]
            + a[7] * I[y + 1, x + 1]
        )
        return r

    def compute_residual_5x5(self, a, I, x, y):
        r = I[y, x] - (
            a[0] * I[y - 2, x - 2]
            + a[1] * I[y - 2, x - 1]
            + a[2] * I[y - 2, x]
            + a[3] * I[y - 2, x + 1]
            + a[4] * I[y - 2, x + 2]
            + a[5] * I[y - 1, x - 2]
            + a[6] * I[y - 1, x - 1]
            + a[7] * I[y - 1, x]
            + a[8] * I[y - 1, x + 1]
            + a[9] * I[y - 1, x + 2]
            + a[10] * I[y, x - 2]
            + a[11] * I[y, x - 1]
            + a[12] * I[y, x + 1]
            + a[13] * I[y, x + 2]
            + a[14] * I[y + 1, x - 2]
            + a[15] * I[y + 1, x - 1]
            + a[16] * I[y + 1, x]
            + a[17] * I[y + 1, x + 1]
            + a[18] * I[y + 1, x + 2]
            + a[19] * I[y + 2, x - 2]
            + a[20] * I[y + 2, x - 1]
            + a[21] * I[y + 2, x]
            + a[22] * I[y + 2, x + 1]
            + a[23] * I[y + 2, x + 2]
        )
        return r


def test_p_map():
    filename = "../lena_1.1.png"
    # filename = "../flower_1.1.png"
    # filename = "../flower_0.9_q101.png"
    # filename = "../../august/suppress_cartoon/arc_0.8.png"
    # filename = "../../august/suppress_cartoon/arc_0.8.png"
    # filename = "../../august/suppress_cartoon/lamp_0.9_q95.png"
    # filename = "../../august/suppress_cartoon/lena_0.9_q95.png"

    detector = ResamplingPopescu()
    
    image = np.random.uniform(0, 255, (600, 600))
    p_map = detector.calculate_probability_map(image)

    print("p_map:")
    print(p_map.shape)
    skimage.io.imsave("p_map.tiff", p_map)
    print("Saved p_map at p_map.tiff")
    
    rescaled_spectrum = detector.calculate_fourier_map(p_map)

    print('rescaled_spectrum')
    print(rescaled_spectrum.shape)
    skimage.io.imsave("rescaled_spectrum.tiff", rescaled_spectrum)
    print("Saved rescaled spectrum at rescaled_spectrum.tiff")


def synthetic_map(height, width, ratio):
    # create a synthetic map related to an affine transform

    # s = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         i0 = np.round(i / ratio) * ratio
    #         j0 = np.round(j / ratio) * ratio

    #         s[i, j] = np.sqrt((i - i0)**2 + (j - j0)**2)

    ii, jj = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ii = ii / ratio
    jj = jj / ratio
    s = np.sqrt( (np.round( ii ) - ii)**2 + (np.round( jj ) - jj) **2 )
    s = s - np.mean(s)
    return s


def test_synthetic_map():
    height = 561
    width = 561
    s = synthetic_map(height, width, ratio=1.1)
    skimage.io.imsave("s.tiff", s)
    print("Saved synthetic map at s.tiff")

    S = np.abs(np.fft.fftn(s, axes=(0, 1), norm="ortho"))
    S = np.fft.fftshift(S, axes=(0, 1))
    skimage.io.imsave("s_fft.tiff", S)
    print("Saved fft of synthetic map at s_fft.tiff")


def generate_synthetic_maps():
    ups_rates = np.arange(0.01, 0.61+1e-10, 0.01)
    downs_rates = np.arange(0.01, 0.41+1e-10, 0.01)

    height, width = 598, 598

    # upsampled synthetic maps
    for r in ups_rates:
        s = synthetic_map(height, width, ratio=1+r)
        s = np.abs(np.fft.fft2(s, axes=(0, 1), norm="ortho"))
        s = np.fft.fftshift(s, axes=(0, 1))
        fname = f"synthetic_maps/S_{(1+r):.3f}.tiff"
        skimage.io.imsave(fname, s)
        print("Saved synthetic map at", fname)
    
    # downsampled synthetic maps
    for r in downs_rates:
        s = synthetic_map(height, width, ratio=1-r)
        s = np.abs(np.fft.fft2(s, axes=(0, 1), norm="ortho"))
        s = np.fft.fftshift(s, axes=(0, 1))
        fname = f"synthetic_maps/S_{(1-r):.3f}.tiff"
        skimage.io.imsave(fname, s)
        print("Saved synthetic map at", fname)


class SearchMap(object):
    def __init__(self, synthetic_map_folder:str) -> None:
        self.folder = synthetic_map_folder
        self.smap_fnames = glob.glob(os.path.join(synthetic_map_folder, "*.tiff"))
        self.smap_fnames.sort()

    def search(self, pmap):
        sim_highest = None
        smap_fname_best = None
        for smap_fname in self.smap_fnames:
            smap = skimage.io.imread(smap_fname)
            # print(smap)
            similarity = np.sum(pmap * smap)

            if sim_highest is None or sim_highest < similarity :
                sim_highest = similarity
                smap_fname_best = smap_fname
        return sim_highest, smap_fname_best
        

def test_search_map():
    searcher = SearchMap(synthetic_map_folder="synthetic_maps")
    detector = ResamplingPopescu()
    # pmap_dump = np.random.uniform(0, 1, (600,600))

    fname = "/Volumes/HPP900/data/resample_detection/target_size_600/r_0.80/bicubic/jpeg_-1/r001d260dt.png"
    img = skimage.io.imread(fname)
    p_map = detector.calculate_probability_map(img)
    P_map = detector.calculate_fourier_map(p_map)
    similarity, smap_fname_best = searcher.search(P_map)


    skimage.io.imsave("P_map.tiff", P_map)
    print("Saved P_map at P_map.tiff")
    
    print(similarity)
    print(smap_fname_best)



def process_one_image(fname:str, detector, searcher):
    img = skimage.io.imread(fname)

    p_map = detector.calculate_probability_map(img)
    P_map = detector.calculate_fourier_map(p_map)

    similarity, smap_fname = searcher.search(P_map)

    print(fname)
    print(similarity)
    print(smap_fname)
    print()

    return fname, ratio, similarity


def main_experiment(ratio, antialias:int):
    # load image
    # compute pmap
    # search smap
    # get lowest score
    target_size = 600
    # ratio = 1.3
    interp = "bicubic"
    jpeg_q1 = -1
    jpeg_q2 = -1

    if antialias == 0:
        input_folder = f"/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        # input_folder = f"/Volumes/HPP900/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        # input_folder = f"/Volumes/HPP900/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
    elif antialias == 1:
        # input_folder = f"/Users/yli/phd/synthetic_image_detection/hongkong/data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        input_folder = f"../../data/resample_detection/target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"
        print(input_folder)
    elif antialias == 2:
        input_folder = f"../../data/resample_detection/matlab/target_size_{target_size}/r_{ratio:.2f}/{interp}_antialias/jpeg_q1{jpeg_q1}/jpeg_q2{jpeg_q2}"

    if jpeg_q2 == -1:
        fname_list = glob.glob(os.path.join(input_folder, "*.png"))
    else:
        fname_list = glob.glob(os.path.join(input_folder, "*.jpg"))

    fname_list.sort()


    detector = ResamplingPopescu()
    searcher = SearchMap(synthetic_map_folder="synthetic_maps")

    df_data = {}
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,similarity
    df_data["fname"] = []
    df_data["target_size"] = []
    df_data["resa_ratio"] = []
    df_data["interp"] = []
    df_data["jpeg_q1"] = []
    df_data["jpeg_q2"] = []
    df_data["similarity"] = []


    from joblib import Parallel, delayed
    from tqdm import tqdm

    num_workers = 32
    results = Parallel(n_jobs=num_workers)(delayed(process_one_image)(fname, detector, searcher) for fname in tqdm(fname_list))

    for fname, ratio, similarity in results:
        df_data["fname"].append(fname)
        df_data["target_size"].append(target_size)
        df_data["resa_ratio"].append(ratio)
        df_data["interp"].append(interp)
        df_data["jpeg_q1"].append(-1)
        df_data["jpeg_q2"].append(-1)
        df_data["similarity"].append(similarity)


    # for fname in tqdm(fname_list):
    #     img = skimage.io.imread(fname)

    #     p_map = detector.calculate_probability_map(img)
    #     P_map = detector.calculate_fourier_map(p_map)


    #     similarity, smap_fname = searcher.search(P_map)
    #     print(smap_fname)

    #     print(fname)
    #     print(similarity)
    #     print()

    #     df_data["fname"].append(fname)
    #     df_data["target_size"].append(target_size)
    #     df_data["resa_ratio"].append(ratio)
    #     df_data["interp"].append(interp)
    #     df_data["jpeg_q1"].append(-1)
    #     df_data["jpeg_q2"].append(-1)
    #     df_data["similarity"].append(similarity)


    # save to .csv
    # fname,target_size,resa_ratio,interp,jpeg_q1,jpeg_q2,similarity
    df = pd.DataFrame(df_data)
    print(df)

    np.set_printoptions(precision=3)

    # output_fname = f"popescu_jpeg_no_resa_ratio_{ratio:.2f}.csv"
    if antialias == 0:  # no anti-alias
        output_fname = f"popescu_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_ratio_{ratio:.2f}.csv"
    elif antialias == 1:  # Gaussian anti-alias
        output_fname = f"popescu_antialias_gaussian_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_ratio_{ratio:.2f}.csv"
    else:  # matlab anti-alias
        output_fname = f"popescu_antialias_matlab_jpeg_q1_{jpeg_q1}_q2_{jpeg_q2}_ratio_{ratio:.2f}.csv"

    df.to_csv(output_fname, index=False, float_format='%4.2f')

    print(f"Saved results at {output_fname}")


if __name__ == "__main__":
    # test_synthetic_map()
    # test_p_map()
    # test_search_map()

    # generate_synthetic_maps()

    print(sys.argv)
    ratio = float(sys.argv[1])
    main_experiment(ratio, antialias=1)
