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
 * @mainpage Chambolle's projection algorithm for TV-denoising
 *
 * README.txt:
 * @verbinclude README.txt
 */


/**
 * @file   chambolle_ipol.cpp
 * @brief  Main executable file
 *
 * @author Joan Duran <joan.duran@uib.es>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libdenoising.h"
#include "libauxiliar.h"
#include "io_png.h"


extern "C" {
    #include "iio.h"
}

// Usage: chambolle_ipol option input.png sigma lambda noisy.png denoised.png

int main(int argc, char **argv)
{
    if(argc < 7) 
	{
		printf("usage: chambolle_ipol option input.png sigma lambda noisy.png "
               "denoised.png\n");
		printf("option       :: 1 :: Add noise to the input image and then "
			                         "denoise with dynamic lambda.\n"
			  "                      - Sigma required from user.\n"
			  "                      - Lambda is set to 0.\n"
			  "                 2 :: Add noise to the input image and then "
			                         "denoise with fixed lambda.\n"
			  "                      - Sigma required from user.\n"
			  "                      - Lambda required from user.\n"
			  "                 3 :: Denoise the input image with dynamic "
                                     "lambda.\n"
		      "                      - Sigma required from user.\n"
	          "                      - Lambda is set to 0.\n"
			  "                      - noisy = input.\n"
			  "                 4 :: Denoise the input image with fixed "
                                     "lambda.\n"
			  "                      - Sigma required from user.\n"
			  "                      - Lambda required from user.\n"
			  "                      - noisy = input.\n");
		printf("input.png    :: input image.\n");
		printf("sigma        :: noise standard deviation.\n");
		printf("lambda       :: trade-off parameter.\n");
		printf("noisy.png    :: noisy image.\n");
		printf("denoised.png :: denoised image by Chambolle's algorithm.\n");
	    
        return EXIT_FAILURE;
	}

	// Read option	
    int option = atoi(argv[1]);

	if(option != 1 && option != 2 && option != 3 && option != 4) 
	{
	    fprintf(stderr, "Error - option must be 1, 2, 3 or 4.\n");
     	return EXIT_FAILURE;
    }

    // Read input image
    size_t nx, ny, nc;
    float *d_v = NULL;

    d_v = io_png_read_f32(argv[2], &nx, &ny, &nc);

    if(!d_v) 
	{
	    fprintf(stderr, "Error - %s not found  or not a correct png image.\n",
                argv[2]);
       	return EXIT_FAILURE;
    }

    // Image variables
	int d_w = (int) nx;
	int d_h = (int) ny;
	int d_c = (int) nc;
	int d_wh = d_w * d_h;
	int d_whc = d_c * d_w * d_h;

    if(d_c == 2)  // We do not use the alpha channel
	    d_c = 1;

   	if(d_c > 3)   // We do not use the alpha channel
	    d_c = 3;

    // Parameters
    float sigma = atof(argv[3]);

    if(sigma < 0.0f)
    {
	    fprintf(stderr, "Error - %s must be nonnegative.\n", argv[3]);
      	return EXIT_FAILURE;
   	} 

    float lambda = 0.0f;

	if(option == 2 || option == 4)        
		lambda = atof(argv[4]);

	if(lambda < 0.0f)
    {
	    fprintf(stderr, "Error - %s must be nonnegative.\n", argv[4]);
       	return EXIT_FAILURE;
    }

    // Add noise to input image if required	
   	float *noisy;

    if(option == 1 || option == 2) 
	{
	    printf("Add noise to the input image and then denoise.\n");
       	noisy = new float[d_whc];

		for(int i = 0; i < d_c; i++) 
	        fiAddNoise(&d_v[i*d_wh], &noisy[i*d_wh], sigma, i, d_wh);
	} else 
	{
		// printf("Denoise the input image.\n");
		noisy = d_v;	
    }

    // TV-denoising
    float **input = new float*[d_c];
    float **output = new float*[d_c];

    for(int c = 0; c < d_c; c++) 
	{
        input[c] = &noisy[c*d_wh];
        output[c] = new float[d_wh];
    }

    if(TVdenoising(option, output, input, sigma, lambda, d_c, d_w, d_h) != 1)
	    return EXIT_FAILURE;

    // Save noisy and denoised as png images
	if(option == 1 || option == 2)
        if(io_png_write_f32(argv[5], noisy, (size_t) d_w, (size_t) d_h,
                            (size_t) d_c) != 0)
		    fprintf(stderr, "Error - Failed to save png image %s.\n", argv[5]);
    
    float *denoised_png = new float[d_whc];
    int k = 0;
	for(int c = 0; c < d_c; c++)
        for(int i = 0; i < d_wh; i++)
        {
            denoised_png[k] = output[c][i];
            k++;
        }
    
    // if(io_png_write_f32(argv[6], denoised_png, (size_t) d_w, (size_t) d_h,
    //                     (size_t) d_c) != 0) 
    //     fprintf(stderr, "Error - Failed to save png image %s.\n", argv[6]);
    iio_write_image_float_vec(argv[6], denoised_png, (size_t) d_w, (size_t) d_h, (size_t) d_c);

	// Free memory
    free(d_v);
    delete[] input;
    
    for(int c = 0; c < d_c; c++)
        delete[] output[c];
    
    delete[] output;
    delete[] denoised_png;
	
    if(option == 1 || option == 2)
        delete[] noisy;

	return EXIT_SUCCESS;
}
