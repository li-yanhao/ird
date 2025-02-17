import numpy as np
import torch
from PIL import Image
import cv2 
from typing import List
import copy


def rgb2luminance(img:np.ndarray):
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 1:
        return img[:,:,0]
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def resize_torch(x:np.ndarray, size, interp):
    if type(size) is int:
        height, width = size, size
    elif type(size) is tuple:
        assert len(size) == 2, f"`size` should be an integer or an tuple of (height, width), got {size}"
        height, width = size

    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize_torch(x[:,:, c], size, interp)
        return x_re

    x = torch.from_numpy(x)[None, None, ...]
    x_re:torch.Tensor = torch.nn.functional.interpolate(x, size=size, mode=interp, antialias=False)
    x_re = x_re[0,0,:,:].numpy()
    return x_re


def test_resize_torch():
    x = np.random.uniform(0,255, size=(40,40,4)).round()
    x_re = resize_torch(x, 49, interp="bicubic")
    print(x_re.shape)


def resize_pil(x:np.ndarray, size, interp):
    # size: int or tuple (height, width)
    # interp: in ["nearest", "bilinear", "bicubic", "lanczos"]

    if type(size) is int:
        height, width = size, size
    elif type(size) is tuple:
        assert len(size) == 2, f"`size` should be an integer or an tuple of (height, width), got {size}"
        height, width = size

    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize_pil(x[:,:, c], size, interp)
        return x_re

    im = Image.fromarray(x)
    
    if interp == "nearest":
        resample_opt = Image.Resampling.NEAREST
    if interp == "bilinear":
        resample_opt = Image.Resampling.BILINEAR
    if interp == "bicubic":
        resample_opt = Image.Resampling.BICUBIC
    if interp == "lanczos":
        resample_opt = Image.Resampling.LANCZOS
    im_resized = im.resize((width, height), resample=resample_opt)

    return np.array(im_resized)


def test_resize_pil():
    x = np.random.uniform(0,255, size=(40,40)).round()
    x_re = resize_pil(x, 49, interp="lanczos")
    print(x_re)


def resize_opencv(image:np.ndarray, size, interp:str):
    if type(size) is int:
        height, width = size, size
    elif type(size) is tuple:
        assert len(size) == 2, f"`size` should be an integer or an tuple of (height, width), got {size}"
        height, width = size

    if interp == "nearest":
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    elif interp == "bicubic":
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    elif interp == "bilinear":
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    elif interp == "lanczos":
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def resize(x:np.ndarray, z=None, size=None, interp="bicubic", antialias=False):
    assert (z is not None) or (size is not None)

    if size is not None:
        if type(size) is int:
            height, width = size, size
        elif type(size) is tuple:
            assert len(size) == 2
            height, width = size

    elif z is not None:
        height, width = x.shape[0:2]
        height, width = round(height * z), round(width * z)
    
    if np.ndim(x) == 3:
        x_re = np.zeros((height, width, x.shape[2]))
        for c in range(x.shape[2]):
            x_re[:,:,c] = resize(x[:,:,c], size=(height, width), interp=interp, antialias=antialias)
        return x_re
    
    if antialias:
        x_re = resize_pil(x, size, interp)
    else:
        x_re = resize_opencv(x, size, interp)

    # if antialias or interp == "lanczos":
    #     x_re = resize_pil(x, size, interp)
    # else:
    #     x_re = resize_torch(x, (height, width), interp)
    return x_re
    


def test_resize():
    x = np.random.uniform(0,255, size=(1000,1000,3)).round()
    for interp in ["nearest", "bilinear", "bicubic", "lanczos"]:
        x_re = resize(x, size=1300, interp=interp, antialias=True)
        x_re = resize(x, size=(1300, 768), interp=interp, antialias=True)
        x_re = resize(x, z=0.76, interp=interp, antialias=True)
        print(x_re.shape)
    for interp in ["nearest", "bilinear", "bicubic"]:
        x_re = resize(x, size=1300, interp=interp, antialias=False)
        x_re = resize(x, size=(1300, 768), interp=interp, antialias=False)
        x_re = resize(x, z=0.76, interp=interp, antialias=False)
        print(x_re.shape)


def ups_2d(x:np.ndarray, z=None, size=None, mode='bicubic', antialias=False):
    """_summary_

    Parameters
    ----------
    x : np.ndarray
        in shape (H, W) or (H, W, C)
    z : _type_, optional
        _description_, by default None
    size : _type_, optional
        _description_, by default None
    mode : str, optional
        _description_, by default 'bicubic'
    antialias : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if np.ndim(x) == 3:
        for c in range(x.shape[2]):
            x[c] = ups_2d(x[..., c], z, size, mode, antialias)
        return x

    assert z is not None or size is not None

    x = torch.from_numpy(x)[None, None, ...]
    if size is not None:
        x = torch.nn.functional.interpolate(x, size=size, mode=mode, antialias=antialias)
    else:
        x = torch.nn.functional.interpolate(x, scale_factor=z, mode=mode, antialias=antialias)
    x = x[0,0,:,:].cpu().numpy()
    return x


def estimate_original_size_jpeg(d_list:list, N:int, rate_range:list, eps:int=2) -> List[int]:
    """Assuming the original image is JPEG-compressed, estimate the original size

    Parameters
    ----------
    d_list : list
        A list of correlation distances {d}
    N : int
        The size of the current image
    eps : int, optional
        range tolerance for avoiding replicated counts, by default 2
    """
    assert len(d_list) > 0

    # step 1: cluster and neighboring suppression
    d_list = np.array(d_list)
    d_list_redundant:np.ndarray = np.concatenate([d_list, N-d_list])
    d_list_redundant.sort()

    d_list_final = []
    d_cluster = [int(d_list_redundant[0])]
    idx = 1
    while idx < len(d_list_redundant):
        if d_list_redundant[idx] - d_cluster[-1] > eps:
            # take the median, and clear the cluster
            d_list_final.append(int(d_cluster[len(d_cluster) // 2]))
            d_cluster = []
        d_cluster.append(int(d_list_redundant[idx]))
        idx += 1
    d_list_final.append(d_cluster[len(d_cluster) // 2])

    # step 2: count the vote for each M
    d_list_final = np.array(d_list_final)
    point_set = []
    for k in range(1, 9):
        for d in d_list_final:
            point_set.append((k,d))
            point_set.append((k,d + N))
            point_set.append((k,d + 2*N))
    M_div8_candidates_ = [d / k for (k,d) in point_set]
    M_div8_candidates_.sort()

    M_div8_candidates = []

    # for rate_range in rate_ranges:
    # for M_div8 in M_div8_candidates_:
        # if N/(M_div8*8) >= rate_range[0] and N/(M_div8*8) < rate_range[1]:
            # M_div8_candidates.append(M_div8)

    for M_div8 in M_div8_candidates_:  # debug x16
        if N/(M_div8*16) >= rate_range[0] and N/(M_div8*16) < rate_range[1]:
            M_div8_candidates.append(M_div8)


    # M/8 * k - q N = d    
    M_div8_best_list = []
    # score_best = 0
    vote_best = 0

    for M_div8 in M_div8_candidates:
        # vote = 0
        map_d_to_kq_list = {d: [] for d in d_list_final}  # {d0: [(k,q),(k,q)], ...} d = k * M/8 + q * N/8  mod  N
        # k_list = []
        # brute force search
        # for k_test in list(range(-8, 9)) + [-16, 16]:
        for k_test in list(range(-16, 17)) + [-32, 32]:  # debug x16
            # for q_test in range(-1, 2):
            for q_test in range(-2,3):
                # d_test = M_div8 * k_test - q_test * N + qq_test * N/8
                d_test = int(M_div8 * k_test + q_test * N/8) % N
                for dd in map_d_to_kq_list.keys():
                    if abs(dd - d_test) <= eps:
                        map_d_to_kq_list[dd].append((k_test, q_test))
        # for k_test in [-16,16]:
        #     d_test = int(k_test * M_div8) % N
        #     # if abs(M_div8 - 197) < 2:
        #     #     print(f"M_div8={M_div8}, d_test={d_test}")
        #     for dd in map_d_to_kq_list.keys():
        #         if abs(dd - d_test) <= eps:
        #             map_d_to_kq_list[dd].append( (k_test, 0) )
        
        # score = sum(compute_kq_list_score(k_list) for k_list in map_d_to_kq_list.values())
        vote = sum(len(kq_list) != 0 for kq_list in map_d_to_kq_list.values())
        score = sum(compute_kq_list_score(kq_list) for kq_list in map_d_to_kq_list.values())
        
        # # score-based
        # if score > score_best:
        #     score_best = score
        #     M_div8_best_list = [(float(M_div8), score)]
        # elif score == score_best:
        #     M_div8_best_list.append((float(M_div8), score))

        # vote-based
        if vote > vote_best:
            vote_best = vote
            # M_div8_best_list = [(float(M_div8), score)]
            M_div8_best_list = [(float(M_div8), score)]
        elif vote == vote_best:
            # M_div8_best_list.append((float(M_div8), score))
            M_div8_best_list.append((float(M_div8), score))


    # print("score_best", score_best)

    # # sort M_div8_best_list:
    # M_div8_best_list = [(M_div8, compute_kq_list_score(k_list)) for M_div8, k_list in M_div8_best_list]

    M_div8_best_list.sort(key=lambda x:x[1], reverse=True)

    for M_div8, score in M_div8_best_list:
        # print(f"M / 8 = {M_div8:.1f}, score = {score:d}")
        # show_matched_points(M_div8, d_list_final, N)

        print(f"M / 16 = {M_div8:.1f}, score = {score:d}")  # debug x16
        show_matched_points(M_div8, d_list_final, N)

    show_matched_points(115.5, d_list_final, N) # debug


    score_best = M_div8_best_list[0][1]
    M_div8_best_list = [pair for pair in M_div8_best_list if pair[1] == score_best]
    # M_list_final = [ int(pair[0] * 8) for pair in  M_div8_best_list]
    M_list_final = [ int(pair[0] * 16) for pair in  M_div8_best_list]  # debug x16

    # print("M_div8_best_list:", M_div8_best_list)
    # print("M_list_final:", M_list_final)

    M_list_final = group_close_value(M_list_final)

    return M_list_final



def estimate_original_size_by_jpegx16(d_list:list, N:int, rate_range:list, eps:int=2) -> List[int]:
    """Assuming the original image is JPEG-compressed, estimate the original size

    Parameters
    ----------
    d_list : list
        A list of correlation distances {d}
    N : int
        The size of the current image
    eps : int, optional
        range tolerance for avoiding replicated counts, by default 2
    """
    assert len(d_list) > 0

    # step 1: cluster and neighboring suppression
    d_list = np.array(d_list)
    d_list_redundant:np.ndarray = np.concatenate([d_list, N-d_list])
    d_list_redundant.sort()

    d_list_final = []
    d_cluster = [int(d_list_redundant[0])]
    idx = 1
    while idx < len(d_list_redundant):
        if d_list_redundant[idx] - d_cluster[-1] > eps:
            # take the median, and clear the cluster
            d_list_final.append(int(d_cluster[len(d_cluster) // 2]))
            d_cluster = []
        d_cluster.append(int(d_list_redundant[idx]))
        idx += 1
    d_list_final.append(d_cluster[len(d_cluster) // 2])

    # step 2: count the vote for each M
    d_list_final = np.array(d_list_final)
    point_set = []
    for k in range(1, 16):
        for d in d_list_final:
            point_set.append((k,d))
            point_set.append((k,d + N))
            point_set.append((k,d + 2*N))
    M_div16_candidates_ = [d / k for (k,d) in point_set]
    M_div16_candidates_.sort()

    M_div16_candidates = []

    for M_div16 in M_div16_candidates_:  # debug x16
        if N/(M_div16*16) >= rate_range[0] and N/(M_div16*16) < rate_range[1]:
            M_div16_candidates.append(M_div16)


    # M/8 * k - q N = d    
    M_div16_best_list = []
    # score_best = 0
    vote_best = 0

    for M_div16 in M_div16_candidates:
        # vote = 0
        map_d_to_kq_list = {d: [] for d in d_list_final}  # {d0: [(k,q),(k,q)], ...} d = k * M/8 + q * N/8  mod  N
        # brute force search
        for k_test in list(range(-16, 17)) + [-32, 32]:  # debug x16
            for q_test in range(-2,3):
                d_test = int(M_div16 * k_test + q_test * N/8) % N
                for dd in map_d_to_kq_list.keys():
                    if abs(dd - d_test) <= eps:
                        map_d_to_kq_list[dd].append((k_test, q_test))

        # for k_test in [-16,16]:
        #     d_test = int(k_test * M_div16) % N
        #     # if abs(M_div16 - 197) < 2:
        #     #     print(f"M_div16={M_div16}, d_test={d_test}")
        #     for dd in map_d_to_kq_list.keys():
        #         if abs(dd - d_test) <= eps:
        #             map_d_to_kq_list[dd].append( (k_test, 0) )
        
        # score = sum(compute_kq_list_score(k_list) for k_list in map_d_to_kq_list.values())
        vote = sum(len(kq_list) != 0 for kq_list in map_d_to_kq_list.values())
        score = sum(compute_kq_list_score_jpegx16(kq_list) for kq_list in map_d_to_kq_list.values())
        
        # # score-based
        # if score > score_best:
        #     score_best = score
        #     M_div16_best_list = [(float(M_div16), score)]
        # elif score == score_best:
        #     M_div16_best_list.append((float(M_div16), score))

        # vote-based
        if vote > vote_best:
            vote_best = vote
            # M_div16_best_list = [(float(M_div16), score)]
            M_div16_best_list = [(float(M_div16), score)]
        elif vote == vote_best:
            # M_div16_best_list.append((float(M_div16), score))
            M_div16_best_list.append((float(M_div16), score))


    # print("score_best", score_best)

    # # sort M_div16_best_list:
    # M_div16_best_list = [(M_div16, compute_kq_list_score(k_list)) for M_div16, k_list in M_div16_best_list]

    M_div16_best_list.sort(key=lambda x:x[1], reverse=True)

    for M_div16, score in M_div16_best_list:
        print(f"M / 16 = {M_div16:.1f}, score = {score:.1f}")  # debug x16
        show_matched_points(M_div16, d_list_final, N)
    # show_matched_points(62.5, d_list_final, N) # debug


    score_best = M_div16_best_list[0][1]
    M_div16_best_list = [pair for pair in M_div16_best_list if pair[1] == score_best]
    M_list_final = [ int(pair[0] * 16) for pair in  M_div16_best_list]  # debug x16

    M_list_final = group_close_value(M_list_final)

    return M_list_final


def compute_kq_list_score(kq_list:list):
    # kq_list_abs = [(abs(k), abs(q)) for k,q in kq_list]
    if len(kq_list) == 0: return 0

    score_map = {
        0: 1000,
        1: 1000,
        2: 100,
        3: 10,
        4: 1,
    }

    k, q = kq_list[0]
    kmin = min(k % 8, (-k) % 8)
    qmin = min(q % 8, (-q) % 8)
    score = score_map[kmin] * 10 + score_map[qmin]
    return score 



def compute_kq_list_score_jpegx16(kq_list:list):
    # kq_list_abs = [(abs(k), abs(q)) for k,q in kq_list]
    if len(kq_list) == 0: return 0

    k_score_map = {
        0: 1000,
        1: 100,
        2: 1000,
        3: 10,
        4: 100,
        5: 1,
        6: 10,
        7: 1,
        8: 1
    }

    q_score_map = {
        0: 1,
        1: 0.5,
        2: 0.5
    }

    best_score = -1
    for k, q in kq_list:
    # k, q = kq_list[0]
        kmin = min(k % 16, (-k) % 16)
        qmin = min(q % 8, (-q) % 8)
        score = k_score_map[kmin] * q_score_map[qmin]
        best_score = best_score if best_score > score else score
    return score 


def group_close_value(l:list, eps=2) -> list:
    if len(l) == 0: return []
    l_final = []
    d_cluster = [copy.deepcopy(l[0])]
    idx = 1
    while idx < len(l):
        if l[idx] - d_cluster[-1] > eps:
            # take the median, and clear the cluster
            l_final.append(copy.deepcopy(d_cluster[len(d_cluster) // 2]))
            d_cluster = []
        d_cluster.append(copy.deepcopy(l[idx]))
        idx += 1
    l_final.append(d_cluster[len(d_cluster) // 2])

    return l_final


def estimate_original_size_non_jpeg(d_list:list, N:int, rate_range:list, eps:int=2):
    d_list = np.array(d_list)
    d_list_redundant:np.ndarray = np.concatenate([d_list, N-d_list])
    d_list_redundant.sort()

    d_list_final = group_close_value(d_list_redundant)

    # step 2: count the vote for each M
    d_list_final = np.array(d_list_final)

    M_candidates_ = []
    # d = kM + qN
    for k in [-2, -1, 1, 2]:
        for d in d_list_final:
            q = 0
            M = d / k
            # while rate_range[0] <= N/M and N/M < rate_range[1]:
            while N/M > 1/2:
                M_candidates_.append(M)
                if k > 0:
                    q = q - 1
                else:
                    q = q + 1
                M = (d - q * N) / k
    
    if len(M_candidates_) == 0:
        return []

    # for k in range(1, 4):
    #     for d in d_list_final:
    #         point_set.append((k,d))
    #         point_set.append((k,d + N))
    #         point_set.append((k,d + 2*N))
    # M_div8_candidates_ = [d / k for (k,d) in point_set]
    M_candidates_.sort()

    map_k_to_score = {
        0: 0,
        1: 1000,
        2: 100,
        3: 10,
    }

    vote_best = -1
    M_and_score_best_list = []
    for M in M_candidates_:
        map_d_to_k = {int(d): 0 for d in d_list_final}
        if N / M >= rate_range[0] and N/ M < rate_range[1]:
            for k in [-2, -1, 1, 2]:
                d_test = (M * k) % N
                for dd in map_d_to_k.keys():
                    if abs(dd - d_test) <= eps:
                        map_d_to_k[dd] = k
    
        vote = sum(k != 0 for k in map_d_to_k.values())
        score = sum(map_k_to_score[abs(k)] for k in map_d_to_k.values())

        # print("M={}, vote={}, score={}, map_d_to_k={}".format(M, vote, score, map_d_to_k))

        # vote-based
        if vote > vote_best:
            vote_best = vote
            M_and_score_best_list = [(M, score)]
        elif vote == vote_best:
            M_and_score_best_list.append((M, score))
    

    M_and_score_best_list.sort(key=lambda k_and_score:k_and_score[1], reverse=True)

    score_best = M_and_score_best_list[0][1]
    M_and_score_best_list = [k_and_score for k_and_score in M_and_score_best_list if k_and_score[1] == score_best]
    M_list_final = [ int(k_and_score[0]) for k_and_score in  M_and_score_best_list ]
    
    M_list_final = group_close_value(M_list_final)

    return M_list_final


def compute_k_list_score(k_list:list):
    if len(k_list) == 0: return 0

    score_map = {
        0: 1000,
        1: 1000,
        2: 100,
        3: 10,
        4: 1,
    }

    k = k_list[0]
    kmin = min(k % 8, (-k) % 8)
    qmin = min(q % 8, (-q) % 8)
    score = score_map[kmin] * 10 + score_map[qmin]
    return score 


def show_matched_points(M_div8, d_list, N, eps=2):
    map_d_to_str = {d: f"? = {d}" for d in d_list}

    print("d_list:", d_list)
    # for k_test in list(range(-8, 9)) + [-16, 16]:
    for k_test in list(range(-16, 17)) + [-32, 32]: # debug x16
        # for q_test in range(-1, 2):
        for q_test in range(-2,3):
            # d_test = M_div8 * k_test - q_test * N + qq_test * N/8
            d_test = int(M_div8 * k_test + q_test * N/8) % N
            for d in d_list:
                if abs(d - d_test) <= eps:
                    printed_str = f"({M_div8:.1f} * {k_test:d} - {q_test:d} * N/8) mod N= {d:d}"
                    map_d_to_str[d] = printed_str
                    break
    
    for d in sorted(d_list):
        print(map_d_to_str[d])

    # for k in range(-8, 9):
    #     for q in range(-1, 2):
    #         for qq in range(-7,8):
    #             d_test = M_div8 * k - q * N + qq * N/8
    #             for dd in d_list:
    #                 if abs(dd - d_test) <= eps: 
    #                     print(f"{M_div8:.1f} * {k:d} - {q:d} * N + {qq:d} * N/8 = {dd:d}")
    # for k in [-2,2]:
    #     d_test = int(k * M_div8 * 8) % N
    #     if abs(M_div8 - 197) < 2:
    #         print(f"M_div8={M_div8}, d_test={d_test}")
    #     for dd in d_list:
    #         if abs(dd - d_test) <= eps:
    #             print(f"{M_div8:.1f} * {k*8:d} + {qq:d} * N/8  mod N = {dd:d}")


def test_estimate_original_size_jpeg():
    d_list = [1, 2,  17, 15, 3,4]
    N = 20
    eps = 1
    M = estimate_original_size_jpeg(d_list, N, eps)
    print("M=", M)


if __name__ == "__main__":
    # test_resize_pil()
    # test_resize_torch()
    # test_resize()
    test_estimate_original_size_jpeg()
