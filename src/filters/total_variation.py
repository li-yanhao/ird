import subprocess
import os
import skimage.io
import numpy as np
import uuid


def tv_extract_noise(img, lmda:float) -> np.ndarray:
    """ Extract noise using Total Variation denoising.
    
    Parameters
    ----------
    img : np.ndarray
        Input image, of shape (H, W).
    lmda : float
        Regularization parameter for TV denoising.
    
    Returns
    -------
    out : np.ndarray
        Extracted noise, of shape (H, W).
    """
    
    # set unique id for each thread
    unique_id = uuid.uuid4()

    input_fname = f"in_{unique_id}.png"
    output_fname = f"out_{unique_id}.tiff"

    # save input image
    skimage.io.imsave(input_fname, img.astype(np.uint8))

    # call the TV denoiser executable
    current_folder = os.path.dirname(os.path.abspath(__file__))
    EXE = os.path.join(current_folder, "CDS_ChambolleTV/chambolle_ipol")
    command = f"{EXE} 4 {input_fname} 10.0 {lmda} noisy.png {output_fname}"
    subprocess.run(command, shell=True)

    # read extracted noise from output file
    out = img - skimage.io.imread(output_fname)

    # clean up temporary files
    os.remove(input_fname)
    os.remove(output_fname)

    return out


def test_tv_extract_noise():

    import numpy as np
    img = np.random.normal(0, 10, (256, 256)) + 128
    out = tv_extract_noise(img, lmda=1)
    print(out)
    print("=== single threading OK === \n")

    from joblib import Parallel, delayed
    import sys
    sys.path.append("..")

    results = Parallel(n_jobs=-1)(delayed(tv_extract_noise)(img, 1) for _ in range(10))
    for result in results:
        print(result)
    print("=== multiple threading OK === \n")


if __name__ == "__main__":
    test_tv_extract_noise()
