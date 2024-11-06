/*
 * Copyright 2009-2013 IPOL Image Processing On Line http://www.ipol.im/
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @mainpage 
 *   - Visualize the difference between two images in such a way that the error
 *     range [-4*sigma, 4*sigma] is linearly transformed to [0, 255] for
 *     visualization purposes. Errors outside this range are saturated to 0
 *     and 255, respectively.
 *   - Compute the Root Mean Squared Error :
 *   			    RMSE = (1/N sum |A[i] - B[i]|^2)^1/2,
 *				    N = number of pixels of the image.
 *   - Compute the Peak Signal-to-Noise Ratio :
 *                      PNSR = 10 log(255^2 / RMSE^2).
 *
 * README.txt:
 * @verbinclude README.txt
 */


/**
 * @file  imdiff_ipol.cpp
 * @brief  Main executable file
 *
 * @author Joan Duran <joan.duran@uib.es>
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "io_png.h"

// Usage: imdiff_ipol image1.png image2.png imdiff.png sigma

int main(int argc, char **argv)
{
    if(argc < 5) 
	{
        printf("usage: imdiff_ipol image1.png image2.png imdiff.png sigma\n");
        printf("image1.png :: first image.\n");
        printf("image2.png :: second image.\n");
        printf("imdiff.png :: difference image.\n");
        printf("sigma      :: noise standard deviation.\n");
	    
        return EXIT_FAILURE;
	}

    // Read first input
    size_t nx, ny, nc;
    float *d_v = NULL;

    d_v = io_png_read_f32(argv[1], &nx, &ny, &nc);

	if (!d_v) 
	{
        fprintf(stderr, "Error - %s not found  or not a correct png image.\n", argv[1]);
        return EXIT_FAILURE;
    }

    if (nc == 2)  // We do not use the alpha channel if grayscale image
	    nc = 1;

    if (nc > 3)   // We do not use the alpha channel if RGB image
	    nc = 3;

    // Read second input
   	size_t nx2, ny2, nc2;
    float *d_v2 = NULL;

	d_v2 = io_png_read_f32(argv[2], &nx2, &ny2, &nc2);

    if (!d_v2) 
	{
	    fprintf(stderr, "Error - %s not found  or not a correct png image.\n",
                argv[2]);
        return EXIT_FAILURE;
    }

    if (nc2 == 2)  // We do not use the alpha channel if grayscale image
	    nc2 = 1;

    if (nc2 > 3)   // We do not use the alpha channel if RGB image
	    nc2 = 3;

    if (nx != nx2 || ny != ny2) 
	{   
        // Check if both images have same size
	    fprintf(stderr, "Error - %s and %s sizes don't match.\n", argv[1],
                argv[2]);
       	return EXIT_FAILURE;
    }

	if (nc != nc2) 
	{
        // Check if both images have same number of channels
		fprintf(stderr, "Error - %s and %s channels don't match.\n", argv[1],
                argv[2]);
       	return EXIT_FAILURE;
    }

    // Image variables
    int d_w = (int) nx;
    int d_h = (int) ny;
    int d_c = (int) nc;
	int d_wh = d_w * d_h;
    int d_whc = d_c * d_wh;

	// Compute RMSE and PSNR
    float fRMSE = 0.0f;
    float fPSNR = 0.0f;

    for(int c = 0; c < d_c ;  c++) 
        for(int i = 0; i < d_wh; i++)
		{
          	float dif = d_v[c*d_wh+i] - d_v2[c*d_wh+i];
            fRMSE += dif * dif;
		}

    fRMSE = fRMSE / (float)d_whc;
    fPSNR = 10.0f * log10f(255.0f * 255.0f / fRMSE);
    fRMSE = sqrtf(fRMSE);

    printf("RMSE: %2.2f\n", fRMSE);
    printf("PSNR: %2.2f\n", fPSNR);

    // Compute image difference. Convert from [-4*sigma, 4*sigma] to [0,255]
    float sigma = atof(argv[4]);    
    sigma *= 4.0f;

   	float *difference = new float[d_whc];

    for(int i = 0; i < d_whc; i++) 
	{
     	float value = d_v[i] - d_v2[i];
     	value =  (value + sigma) * 255.0f / (2.0f * sigma);

        if (value < 0.0)
            value = 0.0f;
        if (value > 255.0)
            value = 255.0f;

        difference[i] = value;
	}

    // Save difference image
    if (io_png_write_f32(argv[3], difference, (size_t) d_w, (size_t) d_h,
                         (size_t) d_c) != 0) 
        fprintf(stderr, "Error - Failed to save png image %s.\n", argv[3]);
    
	// Free memory
	delete[] difference;

    return EXIT_SUCCESS;
}
