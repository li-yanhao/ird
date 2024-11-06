import numpy as np
from numba import jit
import scipy.fftpack
from skimage.util import view_as_windows, view_as_blocks
import scipy.stats


@jit(nopython=True)
def rank_transform(img:np.ndarray, sz:int) -> np.ndarray: # Function is compiled and runs in machine code
    h, w = img.shape
    out = np.zeros((h, w))
    anchor = 0
    for i in range(sz, h-sz):
        for j in range(sz, w-sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[i+di, j+dj] < anchor:
                        rank += 1
                    elif img[i+di, j+dj] == anchor:
                        rank += 0.5
            out[i, j] = rank

    return out




@jit(nopython=True)
def squeezed_rank_transform(img:np.ndarray, sz:int) -> np.ndarray:
    # the zone of rank transform never traverses the 8x8 grid
    h, w = img.shape
    out = np.zeros((h, w))
    anchor = 0
    for i in range(0, h):
        for j in range(0, w):
            rank = 0
            anchor = img[i, j]
            top = max(i - sz, i - i % 8)
            bot = min(i + sz, i - i % 8 + 7)
            left = max(j - sz, j - j % 8)
            right = min(j + sz, j - j % 8 + 7)
            area = (bot - top + 1) * (right - left + 1)
            # if area == 0:
            #     print(f"i,j={i}, {j}")

            for i2 in range(top, bot+1):
                for j2 in range(left, right+1):
                    if img[i2, j2] < anchor:
                        rank += 1
                    elif img[i2, j2] == anchor:
                        rank += 0.5
            out[i, j] = rank / area

    return out

# 1: 0.5
# 2: 0.5 + 1.5 / 1 + 1
# n: n*0.5*n

def rank_transform_no_dc(img:np.ndarray, sz:int) -> np.ndarray:
    # option: remove DC
    h, w = img.shape[0:2]
    blocks = view_as_blocks(img, block_shape=(8,8))
    blocks_dct = scipy.fftpack.dctn(blocks, axes=(-1, -2), norm="ortho")
    blocks_dct[:,:,0,0] = 0
    blocks = scipy.fftpack.idctn(blocks, axes=(-1, -2), norm="ortho")
    img = np.transpose(blocks, axes=(0, 2, 1, 3)).reshape(h, w)

    return rank_transform(img, sz)


def get_rank_to_gaussian_lut(size:int):
    nb_division = size
    rank_to_gaussian = np.zeros(nb_division)
    for i in range(nb_division):
        rank_to_gaussian[i] = scipy.stats.norm.ppf(1/nb_division*(i+.5))

    return rank_to_gaussian


def rank_gaussian_transform(img:np.ndarray, sz:int) -> np.ndarray: # Function is compiled and runs in machine code
    rank_to_gaussian = get_rank_to_gaussian_lut((sz*2+1)*(sz*2+1))
    out = _rank_gaussian_transform_with_lut(img, sz, rank_to_gaussian)
    return out


@jit(nopython=True)
def _rank_gaussian_transform_with_lut(img:np.ndarray, sz:int, rank_to_gaussian:np.ndarray) -> np.ndarray: # Function is compiled and runs in machine code
    h, w = img.shape
    out = np.zeros((h, w))
    anchor = 0

    for i in range(sz, h-sz):
        for j in range(sz, w-sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[i+di, j+dj] < anchor:
                        rank += 1
            out[i, j] = rank_to_gaussian[rank]

    return out


@jit(nopython=True)
def rank_transform_v2(img:np.ndarray, sz:int) -> np.ndarray: # Function is compiled and runs in machine code

    h, w = img.shape

    out = np.zeros((h, w)) # + (sz*2+1) * (sz*2+1) / 2

    anchor = 0
    for i in range(sz, h-sz):
        for j in range(sz, w-sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[i+di, j+dj] < anchor:
                        rank += 1
                    elif img[i+di, j+dj] == anchor:
                        rank += 0.5
            out[i, j] = rank

    for i in range(0, sz):
        for j in range(0, w):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[(i+di) % h, (j+dj) % w] < anchor:
                        rank += 1
                    elif img[(i+di) % h, (j+dj) % w] == anchor:
                        rank += 0.5
            out[i, j] = rank

    for i in range(h-sz, h):
        for j in range(0, w):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[(i+di) % h, (j+dj) % w] < anchor:
                        rank += 1
                    elif img[(i+di) % h, (j+dj) % w] == anchor:
                        rank += 0.5
            out[i, j] = rank
    
    for i in range(sz, h-sz):
        for j in range(0, sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[(i+di) % h, (j+dj) % w] < anchor:
                        rank += 1
                    elif img[(i+di) % h, (j+dj) % w] == anchor:
                        rank += 0.5
            out[i, j] = rank

    for i in range(sz, h-sz):
        for j in range(w-sz, w):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[(i+di) % h, (j+dj) % w] < anchor:
                        rank += 1
                    elif img[(i+di) % h, (j+dj) % w] == anchor:
                        rank += 0.5
            out[i, j] = rank

    return out



@jit(nopython=True)
def accumulate_blocks(blocks:np.ndarray) -> np.ndarray:
    nh, nw, block_sz_h, block_sz_w = blocks.shape[0:4]
    h = nh + block_sz_h - 1
    w = nw + block_sz_w - 1
    out = np.zeros((h, w))
    denom = np.zeros((h, w))
    for i in range(nh):
        for j in range(nw):
            out[i:i+block_sz_h, j:j+block_sz_w] += blocks[i,j,:,:]
            denom[i:i+block_sz_h, j:j+block_sz_w] += 1
    out = out / denom

    return out


def rank_transform_accum(img:np.ndarray, sz:int) -> np.ndarray: # Function is compiled and runs in machine code
    # average the ranks of all the windows containing the target pixel
    h, w = img.shape

    window_sz = 2 * sz + 1
    blocks = view_as_windows(img, (window_sz, window_sz), step=(1, 1))
    nh, nw = blocks.shape[0:2]
    blocks = blocks.reshape(nh, nw, window_sz*window_sz)
    blocks_rank = np.argsort(blocks, axis=-1).reshape(nw, nh, window_sz, window_sz).astype(float)
    # blocks_rank -= (window_sz * window_sz - 1) / 2
    blocks_rank = blocks_rank.astype(int)
    print("blocks_rank")
    print(blocks_rank)

    rank_to_gaussian = get_rank_to_gaussian_lut(window_sz * window_sz)
    blocks_gaussian = rank_to_gaussian[blocks_rank]
    out = accumulate_blocks(blocks_gaussian)

    return out


# @jit(nopython=True)
def rank_transform_align_v0(img:np.ndarray, sz:int) -> np.ndarray: # Function is compiled and runs in machine code

    h, w = img.shape
    assert h % sz ==0 and w % sz == 0
    nh, nw = h // sz, w // sz
    # out = np.zeros((h, w))

    blocks = view_as_windows(img, (sz, sz), step=(sz, sz))
    nh, nw = blocks.shape[0:2]
    blocks = blocks.reshape(nh, nw, sz*sz)
    blocks_rank = np.argsort(blocks, axis=-1).reshape(nw, nh, sz, sz).astype(float)

    out = np.transpose(blocks_rank, axes=(0,2,1,3)).reshape(h, w)
    return out
    
@jit(nopython=True)
def _rank_transform_align(img:np.ndarray) -> np.ndarray: # Function is compiled and runs in machine code
    h, w = img.shape
    assert (h % 8 == 0) and (w % 8 == 0)
    out = np.zeros((h, w))
    anchor = 0
    # n_blk_h, n_blk_w = h // 8, w // 8

    for i in range(h):
        for j in range(w):
            anchor = img[i, j]
            blk_off_i = i - i % 8
            blk_off_j = j - j % 8
            rank = 0
            for i_comp in range(blk_off_i, blk_off_i + 8):
                for j_comp in range(blk_off_j, blk_off_j + 8):
                    if img[i_comp, j_comp] < anchor:
                        rank += 1
                    elif img[i_comp, j_comp] == anchor:
                        rank += 0.5

            out[i, j] = rank

    # for i_blk in range(0, n_blk_h):
    #     for j_blk in range(0, n_blk_w):
    #         for i in range(8):
    #             for j in range(8):

    #         rank = 0
    #         anchor = img[i, j]
    #         for di in range(-sz, sz+1):
    #             for dj in range(-sz, sz+1):
    #                 if img[i+di, j+dj] < anchor:
    #                     rank += 1
    #                 elif img[i+di, j+dj] == anchor:
    #                     rank += 0.5
    #         out[i, j] = rank

    return out


def rank_transform_align_8x8(img_in:np.ndarray, remove_dc:bool) -> np.ndarray:
    # option: remove DC
    if remove_dc:
        h, w = img_in.shape[0:2]
        blocks = view_as_blocks(img_in, block_shape=(8,8))
        blocks_dct = scipy.fftpack.dctn(blocks, axes=(-1, -2), norm="ortho")
        blocks_dct[:,:,0,0] = 0
        blocks = scipy.fftpack.idctn(blocks, axes=(-1, -2), norm="ortho")
        img = np.transpose(blocks, axes=(0, 2, 1, 3)).reshape(h, w)
    else:
        img = img_in.copy()

    return _rank_transform_align(img)


@jit(nopython=True)
def gaussian_kernel(sz, sigma):
    # the kernel is of shape (sz*2+1) x (sz*2+1)
    kernel = np.zeros((sz*2+1, sz*2+1))
    for i in range(sz*2+1):
        for j in range(sz*2+1):
            kernel[i,j] = 1 / (2*np.pi*sigma*sigma) * np.exp(- ((i-sz)*(i-sz) + (j-sz)*(j-sz)) / 2 / sigma / sigma  )
    return kernel


@jit(nopython=True)
def kernel_rank_transform(img:np.ndarray, sz:int, sigma:float) -> np.ndarray: # Function is compiled and runs in machine code
    # how to choose sigma?

    h, w = img.shape

    out = np.zeros((h, w))
    kernel = gaussian_kernel(sz, sigma=sigma)
    print("kernel", kernel)

    anchor = 0
    for i in range(sz, h-sz):
        for j in range(sz, w-sz):
            rank = 0
            anchor = img[i, j]
            for di in range(-sz, sz+1):
                for dj in range(-sz, sz+1):
                    if img[i+di, j+dj] < anchor:
                        rank += 1 * kernel[di + sz, dj + sz]
                    elif img[i+di, j+dj] == anchor:
                        rank += 0.5 * kernel[di + sz, dj + sz]
            out[i, j] = rank

    return out


@jit(nopython=True)
def rank_transform_nxn(img:np.ndarray, n:int) -> np.ndarray:
    # divide an image into non-overlapped nxn blocks, transform each nxn block to one rank value of the top-left pixel
    h, w = img.shape
    assert h % n == 0 and w % n == 0
    h_out, w_out = h // n, w // n
    out = np.zeros((h_out, w_out))
    anchor = 0
    anchor_i, anchor_j = n // 2, n // 2
    for i in range(0, h_out):
        for j in range(0, w_out):
            rank = 0
            anchor = img[i * n + anchor_i, j * n + anchor_j]
            for di in range(0, n):
                for dj in range(0, n):
                    if img[i*n+di, j*n+dj] < anchor:
                        rank += 1
                    elif img[i*n+di, j*n+dj] == anchor:
                        rank += 0.5
            out[i, j] = rank
    return out


    
    

def test_rank_transform():
    # img = np.arange(5*5).reshape(5, 5)
    import iio
    # img = np.random.normal(0, 1, size=(500, 500))
    img = iio.read("/Volumes/HPP900/data/resample_detection/noise_target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/001.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.60/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0b03e77at.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0a71c689t.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q295/r0eb9e900t.jpg")[:,:,0]
    out = rank_transform(img, sz=1)
    print(img)
    print(out)

    iio.write("rt.tiff", out)



def test_kernel_rank_transform():
    # img = np.arange(5*5).reshape(5, 5)
    img = np.random.normal(0, 1, size=(500, 500))
    out = kernel_rank_transform(img, sz=1, sigma=0.3)
    print(img)
    print(out)


def test_rank_gaussian_transform():
    import iio
    img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0b03e77at.jpg")[:,:,0]
    # img = np.arange(5*5).reshape(5, 5)
    # img = np.random.normal(0, 1, size=(500, 500))
    out = rank_gaussian_transform(img, sz=3)
    print(img)
    print(out)
    iio.write("rgt.tiff", out)

def test_rank_transform_accum():
    import iio
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0b03e77at.jpg")[:,:,0]
    # img = np.arange(5*5).reshape(5, 5)
    img = np.random.uniform(0, 1, size=(500, 500))
    out = rank_transform_accum(img, sz=3)
    print(img)
    print(out)
    iio.write("rt_accum.tiff", out)

def test_rank_transform_align():
    import iio
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q295/r0eb9e900t.jpg")[:,:,0]
    h, w = img.shape[0:2]
    img = img[:h-h%8, :w-w%8]

    # print(img.shape)
    # img = np.arange(5*5).reshape(5, 5)
    # img = np.random.uniform(0, 1, size=(500, 500))

    # option: remove DC
    h, w = img.shape[0:2]
    blocks = view_as_blocks(img, block_shape=(8,8))
    blocks_dct = scipy.fftpack.dctn(blocks, axes=(-1, -2), norm="ortho")
    blocks_dct[:,:,0,0] = 0
    blocks = scipy.fftpack.idctn(blocks, axes=(-1, -2), norm="ortho")
    img = np.transpose(blocks, axes=(0, 2, 1, 3)).reshape(h, w)

    out = rank_transform_align_8x8(img)
    # print(img)
    # print(out)
    iio.write("rta.tiff", out)


def test_rank_transform_nxn():
    import iio
    img = np.random.normal(0, 1, size=(500, 500))
    # img = iio.read("/Volumes/HPP900/data/resample_detection/noise_target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/001.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.60/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0b03e77at.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0a71c689t.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q295/r0eb9e900t.jpg")[:,:,0]
    out = rank_transform_nxn(img, n=2)
    iio.write("rt_2x2.tiff", out)



def test_squeezed_rank_transform():
    import iio
    img = np.random.normal(0, 1, size=(500, 500))
    # img = iio.read("/Volumes/HPP900/data/resample_detection/noise_target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/001.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.60/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.00/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q2-1/r0b03e77at.png")[:,:,0]
    # img = iio.read("/Volumes/HPP900/data/resample_detection/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0b03e77at.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q290/r0a71c689t.jpg")[:,:,0]
    # img = iio.read("/Users/yli/phd/synthetic_image_detection/hongkong/integration/data/hard/target_size_600/r_1.40/bicubic/jpeg_q1-1/jpeg_q295/r0eb9e900t.jpg")[:,:,0]
    out = squeezed_rank_transform(img, sz=1)
    iio.write("srt.tiff", out)



if __name__ == "__main__":
    # test_rank_transform()
    # test_kernel_rank_tr/ansform()
    # test_rank_gaussian_transform()
    # test_rank_transform_accum()
    # test_rank_transform_align()
    # test_rank_transform_nxn()
    test_squeezed_rank_transform()

