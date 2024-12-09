import subprocess
import threading
import os

import skimage.io
import numpy as np
import uuid


# WARNING: this cannot be used by multh-threading !
def tv_extract_noise(img, lmda:float, **kwargs) -> np.ndarray:
    
    # thread_id = threading.get_ident()
    # print("thread id:", thread_id)
    unique_id = uuid.uuid4()
    # print("id(img):", id(img))
    # print("unique_id:", unique_id)

    # TODO: wrap TV as an API, with array in array out
    
    input_fname = f"in_{unique_id}.png"
    output_fname = f"out_{unique_id}.tiff"

    skimage.io.imsave(input_fname, img.astype(np.uint8))

    current_folder = os.path.dirname(os.path.abspath(__file__))
    EXE = os.path.join(current_folder, "CDS_ChambolleTV/chambolle_ipol")
    command = f"{EXE} 4 {input_fname} 10.0 {lmda} noisy.png {output_fname}"
    subprocess.run(command, shell=True)

    out = img - skimage.io.imread(output_fname, plugin="pil")
    os.remove(input_fname)
    os.remove(output_fname)
    return out


def test_tv_extract_noise():
    import numpy as np
    img = np.random.normal(0, 1, (256, 256)) + 128
    out = tv_extract_noise(img, lmda=1)
    print(out)
    print("=== single threading OK === \n")

    import sys
    sys.path.append("..")
    from thread_pool_plus import ThreadPoolPlus

    pool = ThreadPoolPlus(workers=2)
    for _ in range(10):
        pool.submit(tv_extract_noise, img, 1)
    while not pool.empty():
        result = pool.pop_result()
        print(result)
    print("=== multiple threading OK === \n")


if __name__ == "__main__":
    test_tv_extract_noise()
