# This script is for plotting tables and figures for experiment part.


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.metrics import roc_curve, roc_auc_score


metric_name = {
   "popescu": "similarity",
   "qiao": "LR",
   "gallagher": "saliency",
   "birajdar": "saliency",
   "mahdian": "saliency",
}

csv_list_uncompressed = {
    "popescu": glob.glob("reimpl/popescu/popescu_jpeg_no_resa_ratio*.csv"),

    "gallagher": glob.glob("reimpl/gallagher/gallagher_jpeg_q1_-1_q2_-1_resa_ratio_*csv"),

    "mahdian": glob.glob("reimpl/mahdian/results/mahdian_jpeg_q1_-1_q2_-1_resa_ratio/*csv"),

    "birajdar": glob.glob("reimpl/birajdar/results_classification/birajdar_jpeg_q1_-1_q2_-1_resa_ratio/birajdar_jpeg_q1_-1_q2_-1_resa_ratio_*.csv"),

    "qiao": glob.glob("/Users/yli/phd/synthetic_image_detection/hongkong/reimpl/qiao/target_600_q1_-1_q2_-1.csv"),

    "ird": glob.glob("/Users/yli/phd/synthetic_image_detection/hongkong/integration/output/1k_impact_proprocess_rt_7x7_correct/*.csv"),

    "ird_cv": glob.glob("raw_result/impact_preprocess_rt_cv/impact_preprocess_rt_cv_q_-1_resa_ratio_*.csv")
}

csv_list_q95 = {
    "popescu": glob.glob("reimpl/popescu/popescu_jpeg_q1_-1_q2_95_*.csv"),

    "gallagher": glob.glob("reimpl/gallagher/gallagher_jpeg_q1_-1_q2_95_resa_ratio_*csv"),

    "mahdian": glob.glob("reimpl/mahdian/results/mahdian_jpeg_q1_-1_q2_95_resa_ratio/*csv"),

    "birajdar": glob.glob("reimpl/birajdar/results_classification/birajdar_jpeg_q1_-1_q2_95_resa_ratio/*csv"),

    "qiao": glob.glob("/Users/yli/phd/synthetic_image_detection/hongkong/reimpl/qiao/target_600_q1_-1_q2_95.csv"),

    "ird": glob.glob("/Users/yli/phd/synthetic_image_detection/hongkong/integration/output/1k_impact_proprocess_q1_-1_q2_95_tv/*csv")
}

csv_list_q90 = {
    "popescu": glob.glob("reimpl/popescu/popescu_jpeg_q1_-1_q2_90_*.csv"),

    "gallagher": glob.glob("reimpl/gallagher/gallagher_jpeg_q1_-1_q2_90_resa_ratio_*csv"),

    "mahdian": glob.glob("reimpl/mahdian/results/mahdian_jpeg_q1_-1_q2_90_resa_ratio/*csv"),

    "birajdar": glob.glob("reimpl/birajdar/results_classification/birajdar_jpeg_q1_-1_q2_90_resa_ratio/*csv"),

    "qiao": glob.glob("reimpl/qiao/target_600_q1_-1_q2_90.csv"),

    "ird": glob.glob("raw_result/1k_impact_proprocess_q1_-1_q2_90_tv/*csv"),

    "ird_cv": glob.glob("raw_result/impact_preprocess_rt_cv/impact_preprocess_rt_cv_q_90_resa_ratio_*.csv")
}

csv_list_interp = {
    "nearest":  glob.glob("raw_result/impact_interpolator/impact_interpolator_nearest_*.csv") ,
    "bilinear": glob.glob("raw_result/impact_interpolator/impact_interpolator_bilinear_*.csv") ,
    "bicubic":  glob.glob("raw_result/impact_interpolator/impact_interpolator_bicubic_*.csv") ,
    "lanczos":  glob.glob("raw_result/impact_interpolator/impact_interpolator_lanczos_*.csv") ,
}

csv_list_image_size = {
    300: glob.glob("raw_result/impact_size/impact_size_300_resa_ratio_*.csv"),
    400: glob.glob("raw_result/impact_size/impact_size_400_resa_ratio_*.csv"),
    500: glob.glob("raw_result/impact_size/impact_size_500_resa_ratio_*.csv"),
    # 600: glob.glob("raw_result/impact_size/impact_size_600_resa_ratio_*.csv"),
    # 600: glob.glob("raw_result/impact_size_600_resa_ratio_*.csv"), # debug
    600: glob.glob("raw_result/impact_interpolator/impact_interpolator_bicubic_*"), # debug
    700: glob.glob("raw_result/impact_size/impact_size_700_resa_ratio_*.csv"),
    800: glob.glob("raw_result/impact_size/impact_size_800_resa_ratio_*.csv"),
}

csv_list_interp_antialias = {
    "bilinear": glob.glob("raw_result/impact_antialias/impact_antialias_bilinear_resa_ratio_*.csv"),
    "bicubic": glob.glob("raw_result/impact_antialias/impact_antialias_bicubic_resa_ratio_*.csv"),
    "lanczos": glob.glob("raw_result/impact_antialias/impact_antialias_lanczos_resa_ratio_*.csv"),
}

csv_list_interp_antialias_matlab = {
    "bilinear": glob.glob("raw_result/impact_antialias_matlab/impact_antialias_matlab_bilinear_resa_ratio_*.csv"),
    "bicubic": glob.glob("raw_result/impact_antialias_matlab/impact_antialias_matlab_bicubic_resa_ratio_*.csv"),
    "lanczos": glob.glob("raw_result/impact_antialias_matlab/impact_antialias_matlab_lanczos_resa_ratio_*.csv"),
}

csv_list_compare_antialias_gaussian = {
    "qiao": glob.glob("reimpl/qiao/target_600_antialias_gaussian_q1_-1_q2_-1.csv"),
    "mahdian": glob.glob("reimpl/mahdian/mahdian_antialias_gaussian_jpeg_q1_-1_q2_-1_resa_ratio_*.csv"),
    "gallagher": glob.glob("reimpl/gallagher/gallagher_antialias_gaussian_jpeg_q1_-1_q2_-1_resa_ratio_*csv"),

    "popescu": glob.glob("reimpl/popescu/popescu_antialias_jpeg_q1_-1_q2_-1_ratio_*.csv"),

    "birajdar": glob.glob("reimpl/birajdar/results_antialias/birajdar_jpeg_q1_-1_q2_-1_antialias_gaussian_resa_ratio_*.csv"),
    "ird": glob.glob("raw_result/impact_antialias/impact_antialias_bicubic_resa_ratio_*.csv"),
    "ird_cv": glob.glob("raw_result/impact_antialias/impact_antialias_bicubic_cv_resa_ratio_*.csv"),
}


csv_list_compare_antialias_matlab = {
    "qiao": glob.glob("reimpl/qiao/target_600_antialias_matlab_q1_-1_q2_-1.csv"),
    "mahdian": glob.glob("reimpl/mahdian/results/mahdian_antialias_matlab_jpeg_q1_-1_q2_-1_resa_ratio/*csv"),
    "gallagher": glob.glob("reimpl/gallagher/gallagher_antialias_matlab_jpeg_q1_-1_q2_-1_resa_ratio_*csv"),

    "popescu": glob.glob("reimpl/popescu/popescu_matlab_jpeg_q1_-1_q2_-1_ratio_*.csv"),
    "birajdar": glob.glob("reimpl/birajdar/results_antialias/birajdar_jpeg_q1_-1_q2_-1_antialias_matlab_resa_ratio_*.csv"),

    "ird": glob.glob("raw_result/impact_antialias_matlab/impact_antialias_matlab_bicubic_resa_ratio_*.csv"),
    "ird_cv": glob.glob("raw_result/impact_antialias_matlab/impact_antialias_matlab_bicubic_cv_resa_ratio_*.csv"),
}


csv_list_preprocess = {
    -1: {

    },
    95: {
        "none": glob.glob("raw_result/impact_preprocess/q_95/impact_preprocess_none_q_95_resa_ratio_*.csv"),
        "phot": glob.glob("raw_result/impact_preprocess/q_95/impact_preprocess_phot_q_95_resa_ratio_*.csv"),
        "dct": glob.glob("raw_result/impact_preprocess/q_95/impact_preprocess_dct_q_95_resa_ratio_*.csv"),
        "tv": glob.glob("raw_result/impact_preprocess/q_95/impact_preprocess_tv_q_95_resa_ratio_*.csv"),
        "rt": glob.glob("raw_result/impact_preprocess/q_95/impact_preprocess_rt_q_95_resa_ratio_*.csv")
    },
    90: {
        "none": glob.glob("raw_result/impact_preprocess/q_90/impact_preprocess_none_q_90_resa_ratio_*.csv"),
        "phot": glob.glob("raw_result/impact_preprocess/q_90/impact_preprocess_phot_q_90_resa_ratio_*.csv"),
        "dct": glob.glob("raw_result/impact_preprocess/q_90/impact_preprocess_dct_q_90_resa_ratio_*.csv"),
        "tv": glob.glob("raw_result/impact_preprocess/q_90/impact_preprocess_tv_q_90_resa_ratio_*.csv"),
        "rt": glob.glob("raw_result/impact_preprocess/q_90/impact_preprocess_rt_q_90_resa_ratio_*.csv")
    }
}  # jpeg_quality -> preprocess_type




def print_info(*text):
    print("\033[1;32m" + " ".join(map(str, text)) + "\033[0;0m")


def _load_compared_method(csv_fname_list, method, metric_name):
    print_info(f"Loaded results of {method}")

    df_list = []
    for csv_fname in csv_fname_list:
        df = pd.read_csv(csv_fname)
        df_list.append(df)
    df =  pd.concat(df_list)

    fpr = 0.01
    df_original = df.query(f'abs(resa_ratio - {1.0}) < 1e-5')
    
    sim_original = np.array(df_original[metric_name])
    sim_original = np.sort(sim_original)[::-1]
    threshold = sim_original[int(fpr * len(sim_original))]

    print(f"At FPR={fpr:.3f}:, threshold={threshold}")
    print()

    recall_list = []
    scores = []
    ratios = df["resa_ratio"].unique()
    ratios = np.sort(ratios)
    # ratios = np.arange(0.6, 1.6+1e-10, 0.1)
    for resa_ratio in ratios:
        df_resampled = df.query(f'abs(resa_ratio - {resa_ratio}) < 1e-5')
        scores_one_ratio = np.array(df_resampled[metric_name])
        recall = (scores_one_ratio >= threshold).mean()
        print(f"recall for resa_ratio={resa_ratio:.3f}: {recall:.3f}")
        recall_list.append(recall)
        scores.append(scores_one_ratio)
    
    print("")
    for e in recall_list: print(e)

    scores = np.stack(scores)
    print()

    return ratios, scores


def load_popescu(csv_fname_list):
    return _load_compared_method(csv_fname_list, method="popescu", metric_name="similarity")


def load_gallagher(csv_fname_list):
    return _load_compared_method(csv_fname_list, method="gallagher", metric_name="saliency")


def load_birajdar(csv_fname_list):
    return _load_compared_method(csv_fname_list, method="birajdar", metric_name="saliency")


def load_mahdian(csv_fname_list):
    return _load_compared_method(csv_fname_list, method="mahdian", metric_name="saliency")


def load_qiao(csv_fname_list):
    return _load_compared_method(csv_fname_list, method="qiao", metric_name="LR")



def iterative_threshold_with_nms(nfa_histos:np.ndarray, target_fpr:float, init_threshold:float, 
                                 anchor_periods:np.ndarray, nms_min_range:int=2) -> float:
    """_summary_

    Parameters
    ----------
    nfa_histos : np.ndarray
        _description_
    target_fpr : float
        _description_
    init_threshold : float
        _description_
    anchor_periods : np.ndarray
        _description_
    nms_min_range : int, optional
        _description_, by default 2

    Returns
    -------
    float
        _description_
    """
    nfa_histos_iter = nfa_histos.copy()
    threshold = init_threshold

    num_periods_jpeg = None
    do_filter = True

    size_histo = nfa_histos.shape[1]
    while do_filter:
        do_filter = False

        # new
        nfa_histos_iter, mask_jpeg = filter_by_nms(nfa_histos_iter, threshold=threshold, anchor_periods=anchor_periods, nms_min_range=nms_min_range)

        # # old
        # positive_mask = (nfa_histos <= threshold)
        # mask_jpeg = np.zeros_like(positive_mask)
        # for entry_img in range(nfa_histos_iter.shape[0]):
        #     for jpeg_period in anchor_periods:
        #         p_left = jpeg_period - nms_min_range - 1
        #         while p_left >= 0 and positive_mask[entry_img, p_left] == True:
        #             mask_jpeg[entry_img, p_left] = True
        #             p_left -= 1
                
        #         p_right = jpeg_period + nms_min_range + 1
        #         while p_right < size_histo and positive_mask[entry_img, p_right] == True:
        #             mask_jpeg[entry_img, p_right] = True
        #             p_right += 1

        # # set the positive mask
        # print(f"removed periods: {np.sum(mask_jpeg)}:")
        # nfa_histos_iter[mask_jpeg] = 1234

        new_num_periods_jpeg = np.sum(mask_jpeg)
        if num_periods_jpeg is None:
            do_filter = True
            num_periods_jpeg = new_num_periods_jpeg
        elif new_num_periods_jpeg > num_periods_jpeg:
            # jpeg periods still exist
            do_filter = True
            num_periods_jpeg = new_num_periods_jpeg
        else:
            do_filter = False
        
        # compute again the threshold
        min_nfas = np.min(nfa_histos_iter, axis=1)
        threshold = np.sort(min_nfas)[int(target_fpr * len(min_nfas))]

        print("new threshold: ", threshold)

    return threshold


def filter_by_nms(nfa_histos, threshold, anchor_periods, nms_min_range:int=2):
    positive_mask = nfa_histos <= threshold

    mask_jpeg = np.zeros_like(positive_mask, dtype=bool)
    nb_histos = nfa_histos.shape[0]
    size_histo = nfa_histos.shape[1]
    for entry_img in range(nb_histos):
        for anchor_period in anchor_periods:
            p_left = anchor_period - nms_min_range - 1
            mask_jpeg[entry_img, p_left+1:anchor_period+1] = True 
            while p_left >= 0 and positive_mask[entry_img, p_left] == True:
                mask_jpeg[entry_img, p_left] = True
                print("fname of suppression: entry_img=", entry_img)
                print(f"nfa[{p_left}]=", nfa_histos[entry_img, p_left])
                p_left -= 1
            
            p_right = anchor_period + nms_min_range + 1
            mask_jpeg[entry_img, anchor_period:p_right] = True
            while p_right < size_histo and positive_mask[entry_img, p_right] == True:
                mask_jpeg[entry_img, p_right] = True
                p_right += 1

    nfa_histos[mask_jpeg] = 1234

    return nfa_histos, mask_jpeg


def load_ird(csv_fname_list, do_nms_jpeg:bool): # ours
    print_info("Our method")

    df_list = []
    for csv_fname in csv_fname_list:
        df = pd.read_csv(csv_fname)
        df_list.append(df)
    df =  pd.concat(df_list)
    print(df)

    df_original = df.query(f'abs(resa_ratio - 1.0) < 1e-5')
    nfa_histos = np.vstack(df_original['nfa'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=" ")))
    # print(df_original.iloc[959]["fname"])

    fpr = 0.01

    min_nfas = np.min(nfa_histos, axis=1)
    threshold = np.sort(min_nfas)[int(fpr * len(min_nfas))]

    current_size = df["target_size"].values[0]
    print("current_size:", current_size)
    max_period = nfa_histos.shape[1]

    if do_nms_jpeg:
        print_info("initial threshold:", threshold)
        jpeg_periods = np.arange(round(current_size / 8), max_period, round(current_size / 8))
        threshold = iterative_threshold_with_nms(nfa_histos, target_fpr=fpr, init_threshold=threshold, anchor_periods=jpeg_periods)
        print_info("new threshold:", threshold)

    print(f"Ours, at FPR={fpr:.3f}: the threshold is {threshold}")
    recall_list = []
    
    # df_false_negative_list = []
    # df_false_positive = None

    scores = []
    # ratios = np.arange(0.6, 1.6+1e-10, 0.1)
    ratios = df["resa_ratio"].unique()
    ratios.sort()
    print("ratios:", ratios)
    for resa_ratio in ratios:

        df_resa_ratio = df.query(f'abs(resa_ratio - {resa_ratio}) < 1e-5')
        nfa_histos = np.vstack(df_resa_ratio['nfa'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=" ")))

        if do_nms_jpeg:
            nfa_histos, _ = filter_by_nms(nfa_histos=nfa_histos, threshold=threshold, anchor_periods=jpeg_periods)

        min_nfa_one_ratio = np.min(nfa_histos, axis=1)

        recall = (min_nfa_one_ratio < threshold).mean()
        print(f"recall for resa_ratio={resa_ratio:.3f}: {recall:.3f}")

        recall_list.append(recall)
        scores.append(-min_nfa_one_ratio) # the nfa is smaller for positive cases

    print("")

    for e in recall_list: print(e)

    # scores = np.stack(scores)
    scores = np.array(scores, dtype=object)
    print()

    return ratios, scores

        # fnames = df_resa_ratio["fname"].to_numpy()

        # # optional: select the not passed cases
        # missing_indexes = np.where(min_nfa_resampled >= threshold)
        # missing_fnames = fnames[missing_indexes]
        # missing_nfas = min_nfa_resampled[missing_indexes]

        # df_FN = df_resa_ratio.loc[missing_indexes]
        # df_FN["nfa_min"] = missing_nfas

        # df_false_negative_list.append(df_FN)

        # if abs(resa_ratio - 1.0) < 1e-5:
        #     fp_indexes = np.where(min_nfa_resampled < threshold)
        #     df_false_positive = df_resa_ratio.iloc[fp_indexes]
        #     fp_nfas = min_nfa_resampled[fp_indexes]
        #     df_false_positive["nfa_min"] = fp_nfas


        # # TODO: compute the false positive periods and true positive periods
        # prev_size = current_size / resa_ratio
        # true_periods = calculate_secondary_periods_by_jpeg(primary=prev_size, total_size=current_size)
        # print("true_periods:", true_periods)

        # count_tp = 0
        # count_fp = 0
        # for img_entry in range(nfa_histos.shape[0]):
        #     positive_periods = np.argwhere(nfa_histos[img_entry] < threshold).flatten()
        #     tp_mask = np.isin(positive_periods, true_periods)
        #     # print("positive_periods:", positive_periods)
        #     # print("tp_mask:", tp_mask)
        #     count_tp += np.any(tp_mask)
        #     count_fp += np.any(~tp_mask)

        # print()
        # print(f"At resa_ratio={resa_ratio:.2f}")
        # print(f"images containing TP periods: {count_tp/999:.3f}")
        # print(f"images containing FP periods: {count_fp/999:.3f}")
        # print()

        # positive_mask = nfa_histos < threshold
        # mask_jpeg = np.zeros_like(positive_mask, dtype=bool)
        # for entry_img in range(nfa_histos.shape[0]):
        #     for jpeg_period in jpeg_periods:
        #         p_left = jpeg_period - 3
        #         mask_jpeg[p_left+1:jpeg_period+1] = True 
        #         while p_left >= 0 and positive_mask[entry_img, p_left] == True:
        #             mask_jpeg[entry_img, p_left] = True
        #             p_left -= 1
                
        #         p_right = jpeg_period + 3
        #         mask_jpeg[jpeg_period:p_right] = True
        #         while p_right < max_period and positive_mask[entry_img, p_right] == True:
        #             mask_jpeg[entry_img, p_right] = True
        #             p_right += 1

        # nfa_histos[mask_jpeg] = 1
        # # compute again the threshold
        # min_nfa_resampled = np.min(nfa_histos, axis=1)



        # recall = (min_nfa_resampled < threshold).mean()
        # print(f"recall for resa_ratio={resa_ratio:.3f}: {recall:.3f}")
        # recall_list.append(recall)

        # # print false negative cases
        # df_resa_rate = df.query(f'abs(resa_ratio - {resa_ratio}) < 1e-5')
        # fnames = df_resa_rate["fname"].to_numpy()
        # sorted_idx = np.argsort(min_nfa_resampled)
        # min_nfa_resampled = min_nfa_resampled[sorted_idx]
        # fnames = fnames[sorted_idx]
        # print("False negative cases:")
        # for i in range(int(fpr * len(min_nfa_resampled))):
        #     print(fnames[-(i+1)])
        #     print(min_nfa_resampled[-(i+1)])

    # for e in recall_list: print(e)


    # df_false_negative = pd.concat(df_false_negative_list)
    # csv_fname = "false_negative_cases.csv"
    # df_false_negative = df_false_negative.drop('nfa', axis=1)
    # df_false_negative.to_csv(csv_fname, index=False)
    # print(f"Saved the false negative cases in {csv_fname}")

    # csv_fname = "false_positive_cases.csv"
    # df_false_positive = df_false_positive.drop('nfa', axis=1)
    # df_false_positive.to_csv(csv_fname, index=False)
    # print(f"Saved the false positive cases in {csv_fname}")

    
    # return ratios, scores


def print_accuracies():

    # load_birajdar(csv_list_uncompressed["birajdar"])
    # load_birajdar(csv_list_q95["birajdar"])
    # load_birajdar(csv_list_q90["birajdar"])

    # load_gallagher(csv_list_uncompressed["gallagher"])
    # load_gallagher(csv_list_q95["gallagher"])
    # load_gallagher(csv_list_q90["gallagher"])

    # load_mahdian(csv_list_uncompressed["mahdian"])
    # load_mahdian(csv_list_q95["mahdian"])
    # load_mahdian(csv_list_q90["mahdian"])

    # load_qiao(csv_list_uncompressed["qiao"])
    # load_qiao(csv_list_q95["qiao"])
    # load_qiao(csv_list_q90["qiao"])

    # load_popescu(csv_list_uncompressed["popescu"])
    # load_popescu(csv_list_q95["popescu"])
    # load_popescu(csv_list_q90["popescu"])

    # load_ird(csv_list_uncompressed["ird"], do_nms_jpeg=False)  # ok
    load_ird(csv_list_q95["ird"], do_nms_jpeg=True)  # ok
    # load_ird(csv_list_q90["ird"], do_nms_jpeg=True)  # ok



def plot_roc_curves_comparison(q:int, antialias:int):
    """_summary_

    Parameters
    ----------
    q : int
        jpeg quality, in {-1, 95, 90}
    """
    if q == -1:
        csv_list = csv_list_uncompressed
        do_nms_jpeg = False
    elif q == 95:
        csv_list = csv_list_q95
        do_nms_jpeg = True
    elif q == 90:
        csv_list = csv_list_q90
        do_nms_jpeg = True


    if antialias == 0:
        method_scores = {
            "Qiao": load_qiao(csv_list["qiao"]),
            "Mahdian": load_mahdian(csv_list["mahdian"]),
            "Gallagher": load_gallagher(csv_list["gallagher"]), 
            "Birajdar": load_birajdar(csv_list["birajdar"]), 
            "Popescu": load_popescu(csv_list["popescu"]),
            "Ours": load_ird(csv_list["ird"], do_nms_jpeg),
            "Ours, c.v.": load_ird(csv_list["ird_cv"], do_nms_jpeg),
        }

    if antialias == 1:
        method_scores = {
            "Qiao": load_qiao(csv_list_compare_antialias_gaussian["qiao"]),
            "Mahdian": load_mahdian(csv_list_compare_antialias_gaussian["mahdian"]),
            "Gallagher": load_gallagher(csv_list_compare_antialias_gaussian["gallagher"]), 
            "Birajdar": load_birajdar(csv_list_compare_antialias_gaussian["birajdar"]), 
            "Popescu": load_popescu(csv_list_compare_antialias_gaussian["popescu"]),
            "Ours": load_ird(csv_list_compare_antialias_gaussian["ird"], do_nms_jpeg=do_nms_jpeg),
            "Ours, c.v.": load_ird(csv_list_compare_antialias_gaussian["ird_cv"], do_nms_jpeg=do_nms_jpeg),
        }

    if antialias == 2:
        method_scores = {
            "Qiao": load_qiao(csv_list_compare_antialias_matlab["qiao"]),
            "Mahdian": load_mahdian(csv_list_compare_antialias_matlab["mahdian"]),
            "Gallagher": load_gallagher(csv_list_compare_antialias_matlab["gallagher"]), 
            "Birajdar": load_birajdar(csv_list_compare_antialias_matlab["birajdar"]), 
            "Popescu": load_popescu(csv_list_compare_antialias_matlab["popescu"]),
            "Ours": load_ird(csv_list_compare_antialias_matlab["ird"], do_nms_jpeg=do_nms_jpeg),
            "Ours, c.v.": load_ird(csv_list_compare_antialias_matlab["ird_cv"], do_nms_jpeg=do_nms_jpeg),
        }


    for method in method_scores.keys():

        ratios, scores = method_scores[method]

        mask_entry_neg = np.abs(ratios - 1.0) < 1e-5

        if q == -1:
            mask_entry_pos = np.abs(ratios - 1.0) > 1e-5
        else:
            mask_entry_pos = np.logical_and.reduce((
                np.abs(ratios - 0.8) > 1e-5,
                np.abs(ratios - 1.0) > 1e-5,
                np.abs(ratios - 1.6) > 1e-5
            ))

        scores_neg = scores[mask_entry_neg].flatten()
        scores_pos = scores[mask_entry_pos].flatten()
        scores = np.concatenate([scores_neg, scores_pos])
        y = [0] * len(scores_neg) + [1] * len(scores_pos)
        # print(scores_neg.astype(float))
        scores = scores.astype(float)
        scores[np.isinf(scores)] = 1e10
        scores[np.isnan(scores)] = 0
        print("method:", method)
        print("scores:", scores.shape)
        print("y:", len(y))

        fpr, tpr, thresholds = roc_curve(y, scores)
        roc_auc = roc_auc_score(y, scores)
        if method.lower() in ["ours", "ours, c.v."]:
            plt.plot(fpr, tpr, "--", label=f"{method}: auc={roc_auc:.3f}")
        else:
            plt.plot(fpr, tpr, "-", label=f"{method}: auc={roc_auc:.3f}")

    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.show()




def plot_ablation_preprocess(q:int):
    preprocess_list = ["none", "phot", "dct", "tv", "rt"]
    fpr = 0.01
    if q > 0:
        do_nms_jpeg = True
    else:
        do_nms_jpeg = False

    for preprocess in preprocess_list:
        print_info(f"preprocess: {preprocess}")
        ratios, scores_ratios = load_ird(csv_fname_list=csv_list_preprocess[q][preprocess], do_nms_jpeg=do_nms_jpeg)
        scores_original = scores_ratios[np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0]]
        scores_original = np.sort(scores_original)
        threshold = scores_original[-int(fpr * len(scores_original))]

        if q <= 0:
            ratio_valid_mask = np.abs(ratios - 1) > 1e-5
        else:
            ratio_valid_mask = np.logical_and.reduce((
                np.abs(ratios - 1) > 1e-5,
                np.abs(ratios - 0.8) > 1e-5,
                np.abs(ratios - 1.6) > 1e-5
            ))  

        # if q <= 0:
        scores_ratios_valid = scores_ratios[ratio_valid_mask]
        
        ratios_valid = ratios[ratio_valid_mask]
        scores_ratios_valid = scores_ratios[ratio_valid_mask]

        acc_list = np.zeros(len(ratios_valid))
        print()
        print("threshold:", threshold)
        for i, ratio in enumerate(ratios_valid):
            if q <= 0 and (abs(ratio - 1) < 1e-5):
                acc_list[i] = 0
                continue
            if q > 0 and (abs(ratio - 1) < 1e-5 or abs(ratio - 0.8) < 1e-5 or abs(ratio - 1.6) < 1e-5):
                acc_list[i] = 0
                continue
            scores_one_ratio = scores_ratios_valid[i]
            acc = (scores_one_ratio > threshold).mean()
            acc_list[i] = acc
            print(f"resa_ratio={ratio:.2f}, acc={acc}")
        acc_list = np.array(acc_list)

        # ratios_to_plot = ratios[~np.isnan(acc_list)]
        # acc_to_plot = acc_list[~np.isnan(acc_list)] * 100
        plt.plot(ratios_valid, acc_list * 100, marker="o", label=f"{preprocess}")
    
    plt.legend()
    plt.xlabel("Resampling factor")
    plt.ylabel("Accuracy (%)")
    plt.xticks(ratios)
    plt.tight_layout()
    plt.show()


def plot_ablation_size():
    size_list = [300, 400, 500, 600]
    # size_list = [300, 400, 500, 600, 700, 800]
    # size_list = [600]

    # additional step: use pre-fixed ratios
    fixed_ratios = np.arange(0.6, 2.0+1e-5, 0.1)


    fpr = 0.01
    for size in size_list:
        print_info(f"size: {size}")
        ratios, scores_ratios = load_ird(csv_fname_list=csv_list_image_size[size], do_nms_jpeg=False)

        if fixed_ratios is not None:
            print("fixed_ratios", fixed_ratios)
            print("ratios", ratios)
            print("[ np.min(r - fixed_ratios) < 1e-5 for r in ratios ]", [ np.abs(np.min(r - fixed_ratios)) < 1e-5 for r in ratios ])
            scores_ratios = scores_ratios[[ np.min(np.abs(r - fixed_ratios)) < 1e-5 for r in ratios ]]
            ratios = fixed_ratios
        print("scores_ratios:", len(scores_ratios))
        scores_original = scores_ratios[np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0]]
        scores_original = np.sort(scores_original)
        threshold = scores_original[-int(fpr * len(scores_original))]

        # ratio_valid_mask = np.abs(ratios - 1) > 1e-5
        # scores_ratios_valid = scores_ratios[ratio_valid_mask]
        
        # ratios_valid = ratios[ratio_valid_mask]
        # scores_ratios_valid = scores_ratios[ratio_valid_mask]

        acc_list = np.zeros(len(ratios))
        print()
        print("threshold:", threshold)
        for i, ratio in enumerate(ratios):
            if (abs(ratio - 1) < 1e-5):
                acc_list[i] = 0
                continue
            scores_one_ratio = scores_ratios[i]
            acc = (scores_one_ratio > threshold).mean()
            acc_list[i] = acc
            print(f"resa_ratio={ratio:.2f}, acc={acc}")
        acc_list = np.array(acc_list)

        # ratios_to_plot = ratios[~np.isnan(acc_list)]
        # acc_to_plot = acc_list[~np.isnan(acc_list)] * 100
        plt.plot(ratios, acc_list * 100, marker="o", label=f"{size}x{size}")
    
    plt.legend()
    plt.xlabel("Resampling factor")
    plt.ylabel("Accuracy (%)")
    plt.xticks(ratios)
    plt.tight_layout()
    plt.show()
    pass


def tutorial_zoomin():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Sample data
    x = range(100)
    y1 = [i**0.5 for i in x]         # First curve
    y2 = [i**0.5 + 1 for i in x]     # Second curve (offset)
    y3 = [i**0.5 - 1 for i in x]     # Third curve (offset)

    # Create the main plot
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Curve 1')
    ax.plot(x, y2, label='Curve 2')
    ax.plot(x, y3, label='Curve 3')
    ax.legend()

    # Create an inset axis on top of the main plot
    inset_ax = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=5)  # Adjust position

    # Plot the same curves in the inset, focusing on a specific x-range
    inset_ax.plot(x, y1, label='Curve 1')
    inset_ax.plot(x, y2, label='Curve 2')
    inset_ax.plot(x, y3, label='Curve 3')

    # Set the x and y limits of the inset to zoom into the area of interest
    inset_ax.set_xlim(30, 50)  # Zoomed x-axis range
    inset_ax.set_ylim(4, 7.5)  # Zoomed y-axis range

    # Optionally, remove legend or labels from the inset if they overlap or clutter the view
    # inset_ax.get_legend().remove()  # Remove legend in the inset, if desired

    # Add main axis labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    plt.title("Main Plot with Zoomed Inset")

    # Save the figure
    plt.savefig("zoomed_multi_curve_plot.pdf", format="pdf")
    plt.show()



def plot_ablation_interpolator():
    interpolators = ["nearest", "bilinear", "bicubic", "lanczos"]
    # interpolators = ["bicubic"]
    fixed_ratios = np.concatenate([
        np.arange(0.6, 0.9, 0.1),
        np.arange(0.9, 0.961, 0.01),
        np.array([1]),
        np.arange(1.04, 1.1, 0.01),
        np.arange(1.1, 2.0+1e-5, 0.1),
    ])
    fpr = 0.01

    fig, ax = plt.subplots()
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=3)  # Adjust as needed

    for interp in interpolators:
        ratios, scores_ratios = load_ird(csv_fname_list=csv_list_interp[interp], do_nms_jpeg=False)

        ratio_to_scores = {}
        for ratio, scores in zip(ratios, scores_ratios):
            ratio_to_scores[round(ratio, 4)] = scores

        acc_list = np.zeros(len(fixed_ratios))
        print()
        scores_original = scores_ratios[np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0]]
        # print("np.argwhere(np.abs(ratios - 1) < 1e-5)[0]:", np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0])
        # print("scores_original:", scores_original)
        scores_original = np.sort(scores_original)
        threshold = scores_original[-int(fpr * len(scores_original))]
        print("threshold:", threshold)

        for i, ratio in enumerate(fixed_ratios):
            if abs(round(ratio, 4) - 1) < 0.035:
                acc_list[i] = 0
                continue
            scores_one_ratio = ratio_to_scores[round(ratio, 4)]

            acc = (scores_one_ratio > threshold).mean()
            acc_list[i] = acc
            print(f"resa_ratio={ratio:.2f}, acc={acc}")
        acc_list = np.array(acc_list)

        ratios_to_plot = fixed_ratios[~np.isnan(acc_list)]
        acc_to_plot = acc_list[~np.isnan(acc_list)] * 100
        ax.plot(ratios_to_plot, acc_to_plot, marker="o", markersize=4, label=f"{interp}")
        inset_ax.plot(ratios_to_plot, acc_to_plot, marker="o", markersize=4, label=f"{interp}")
    
    ax.legend()

    

    # Set the limits of the inset to zoom into a specific part of the data
    
    # inset_ax.set_xticks(np.arange(0.9, 1.10001, 0.01))
    inset_ax.set_xlim(0.9, 1.1)  # X-axis range for zoomed-in area
    inset_ax.set_ylim(56, 102)  # Y-axis range for zoomed-in area
    # inset_ax.get_legend().remove()  
    
    ax.set_xlabel("Resampling factor")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(np.arange(0.6, 2.001, 0.1))
    plt.tight_layout()
    plt.show()



def plot_ablation_antialias(use_matlab:bool):
    # interpolators = ["nearest", "bilinear", "bicubic", "lanczos"]
    interpolators = ["bilinear", "bicubic", "lanczos"]
    fixed_ratios = np.arange(0.6, 1.0001, 0.05)
    fpr = 0.01

    fig, ax = plt.subplots()
    # inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=3)  # Adjust as needed

    for interp in interpolators:
        print_info("interp:", interp)
        if use_matlab:
            ratios, scores_ratios = load_ird(csv_fname_list=csv_list_interp_antialias_matlab[interp], do_nms_jpeg=False)
        else:
            ratios, scores_ratios = load_ird(csv_fname_list=csv_list_interp_antialias[interp], do_nms_jpeg=False)

        ratio_to_scores = {}
        for ratio, scores in zip(ratios, scores_ratios):
            ratio_to_scores[round(ratio, 4)] = scores

        acc_list = np.zeros(len(fixed_ratios))
        print()
        scores_original = scores_ratios[np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0]]
        # print("np.argwhere(np.abs(ratios - 1) < 1e-5)[0]:", np.argwhere(np.abs(ratios - 1) < 1e-5)[0,0])
        # print("scores_original:", scores_original)
        scores_original = np.sort(scores_original)
        threshold = scores_original[-int(fpr * len(scores_original))]
        print("threshold:", threshold)

        for i, ratio in enumerate(fixed_ratios):
            if abs(round(ratio, 4) - 1) < 0.035:
                acc_list[i] = 0
                continue
            scores_one_ratio = ratio_to_scores[round(ratio, 4)]

            acc = (scores_one_ratio > threshold).mean()
            acc_list[i] = acc
            print(f"resa_ratio={ratio:.2f}, acc={acc}")
        acc_list = np.array(acc_list)

        ratios_to_plot = fixed_ratios[~np.isnan(acc_list)]
        acc_to_plot = acc_list[~np.isnan(acc_list)] * 100
        ax.plot(ratios_to_plot, acc_to_plot, marker="o", markersize=4, label=f"{interp} + antialias")
        # inset_ax.plot(ratios_to_plot, acc_to_plot, marker="o", markersize=4, label=f"{interp}")
    
    ax.legend()


    # Set the limits of the inset to zoom into a specific part of the data
    
    # inset_ax.set_xticks(np.arange(0.9, 1.10001, 0.01))
    # inset_ax.set_xlim(0.9, 1.1)  # X-axis range for zoomed-in area
    # inset_ax.set_ylim(56, 102)  # Y-axis range for zoomed-in area
    # inset_ax.get_legend().remove()  
    
    ax.set_xlabel("Resampling factor")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(np.arange(0.6, 1.0001, 0.05))
    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    # print_accuracies()
    plot_roc_curves_comparison(q=90, antialias=False)
    # plot_ablation_preprocess(q=90)
    # plot_ablation_size()
    # tutorial_zoomin()
    # plot_ablation_interpolator()

    # plot_ablation_antialias(use_matlab=False)

    # plot_roc_curves_comparison(q=-1, antialias=1)
