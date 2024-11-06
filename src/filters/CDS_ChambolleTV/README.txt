Chambolle's Projection Algorithm for Total Variation Denoising

Joan Duran, joan.duran@uib.es, Universitat de les Illes Balears (Spain)
Bartomeu Coll, tomeu.coll@uib.es, Universitat de les Illes Balears (Spain)
Catalina Sbert, catalina.sbert@uib.es, Universitat de les Illes Balears (Spain)


# OVERVIEW

This C source code accompanies with Image Processing On Line (IPOL) article
"Chambolle's Projection Algorithm for Total Variation Denoising" at 

    http://www.ipol.im/pub/pre/61/

This code is used by the online IPOL demo:

    http://dev.ipol.im/~lisani/ipol_demo/cds_chambolleTV/

This program reads and writes PNG images, but can be easily adapted to any
other file format. Only 8bit PNG images are handled. 

Two programs are provided:

* 'chambolle_ipol' reads a noise-free png image, adds white Gaussian noise and
then denoises it using Chambolle's projection algorithm with dynamic or fixed
lambda parameter. It also permits to read a noisy png image and denoises it with
dynamic or fixed lambda parameter. In all cases, the noise standard deviation
is required. If user introduces a noisy image with very wrong estimation of
sigma, the algorithm probably needs more iterations to converge. In this case,
we suggest to download the source code and execute the algorithm with less
tolerance, for instance, 0.0001.

* 'imdiff_ipol' visualizes the difference between two images in such a way that
the error range is linearly transformed from [-4*sigma, 4*sigma] to [0, 255].
Errors outside this range are saturated to 0 and 255, respectively. It also
computes the Root Mean Squared Error and the Peak Signal-to-Noise Ratio:    
    - RMSE = (1/N sum |A[i] - B[i]|^2)^1/2,
    - PNSR = 10 log(255^2 / RMSE^2).


# USAGE

Usage: chambolle_ipol option input.png sigma lambda noisy.png denoised.png

option       :: 1 :: Add noise to the input image and then denoise with dynamic
                     lambda. Sigma is required from user and lambda is set to 0.
                2 :: Add noise to the input image and then denoised with fixed
                     lambda. Sigma and lambda are required from user.
                3 :: Denoise the input image with dynamic lambda. Sigma is
                     required from user, lambda is set to 0 and noisy = input.
                4 :: Denoise the input image with fixed lambda. Sigma and
                     lambda are required from user and noisy = input.
input.png    :: input image.
sigma        :: noise standard deviation.
lambda       :: trade-off parameter.
noisy.png    :: noisy image.
denoised.png :: denoised image by Chambolle's projection algorithm.

If dynamic lambda has been selected, this program also provides on screen the
values of lambda tuning.

Usage: imdiff_ipol image1.png image2.png imdiff.png sigma 

image1.png : first image.
image2.png : second image.
imdiff.png : difference image.
sigma      : noise standard deviation.

This program also provides on screen the RMSE and the PSNR values.


#LICENSE

Files mt199937.ar.c and mt19937.ar.h are copyright Makoto Matsumoto and 
Takuji Nishimura. Files io_png.c and io_png.h are copyright Nicolas Limare.
These files are distributed under the BSD license conditions described in
the corresponding headers files.

All the other files are distributed under the terms of the GNU General
Public License Version 3 (the "GPL").



# REQUIREMENTS

The code is written in ANSI C and C++, and should compile on any system 
with an ANSI C/C++ compiler.

The libpng header and libraries are required on the system for compilation
and execution. 

The implementation uses OPENMP which not supported by old versions of
gcc (older than the 4.2). 


# COMPILATION

Simply use the provided makefile, with the command 'make'.


# EXAMPLE
OPTION 1 : Add noise to the input image and then denoise with dynamic lambda
    
    option=1    
    input=traffic.png
    sigma=10.0
    lambda=0.0
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma

OPTION 2 : Add noise to the input image and then denoise with fixed lambda
    
    option=2    
    input=traffic.png
    sigma=10.0
    lambda=0.08
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma

OPTION 3 : Denoise noisy image with dynamic lambda
    
    option=3   
    input=traffic_noisy.png
    sigma=10.0
    lambda=0.0
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma

OPTION 4 : Denoise noisy image with fixed lambda
     
    option=4    
    input=traffic_noisy.png
    sigma=10.0
    lambda=0.08
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma
