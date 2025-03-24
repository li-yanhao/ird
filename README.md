# Image Resampling Detection via Spectral Correlation and False Alarm Control


Image resampling exhibits correlations in the Fourier spectrum that are exploitable for image resampling detection.
We propose an unsupervised _a contrario_ method to detect anomalous spectral correlations. 
The proposed method is suitable for images resampled with different anti-aliasing filters, scaling factors and interpolating filters and post-compressed at different levels.

📄 [Preprint](https://doi.org/10.36227/techrxiv.174235422.29936668/v1)

🚀 [Demo](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000537)


## Prepare

Compile the TV denoiser in `src/third_party`:
``` bash
cd src/filters/CDS_ChambolleTV
make
```

## Test on a single image
``` bash
python test/test_one_image.py <img_fname>
```

