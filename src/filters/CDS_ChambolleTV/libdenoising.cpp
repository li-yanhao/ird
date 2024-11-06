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
 * @file libdenoising.cpp
 * @brief functions for TV-denoising with Chambolle's projection algorithm.
 * @author Joan Duran <joan.duran@uib.es>
 */

#include "libdenoising.h"

/**
 * \brief  Compute discrete gradient operator via forward differences.
 *
 * @param[in]  u  input vector : the first pointer accounts for the channel and
 *             the second one for the pixel position.
 * @param[out] grad  gradient operator : the first pointer accounts for the
 *             channel, the second one for the directional derivative (u_x or
 *             u_y) and the third one for the pixel position.
 * @param[in]  channels  number of channels of the image.
 * @param[in]  width, height  image size.
 *
 */

void gradient(float **u, float ***grad, int num_channels, int width,
              int height)
{
    for(int c = 0; c < num_channels; c++)
	{
		for(int j = 0; j < height; j++) 
		{
			int k = j*width;
			
			for(int i = 0; i < width; i++)
			{
				// Derivatives in the x-direction
				if(i != width-1) 
					grad[c][0][i+k] = u[c][i+1+k] - u[c][i+k];
				else 
					grad[c][0][i+k] = 0.0f;
				
				// Derivatives in the y-direction
				if(j != height-1) 
					grad[c][1][i+k] = u[c][i+width+k] - u[c][i+k];
				else 
					grad[c][1][i+k] = 0.0f;
			}	
		}
	}	
}

/**
 * \brief  Compute divergence operator as @f$ \langle -\mbox{div} p, u \rangle =
 *         \langle p, \nabla  u\rangle @f$.
 *
 * @param[in]  p  dual variable : the first pointer accounts for the channel,
 *             the second one for the coordinate (x or y) and the third one for
 *             the pixel position.
 * @param[out] div_p  divergence operator : the first pointer accounts for the
 *             channel and the second one for the pixel position.
 * @param[in]  channels  number of channels of the image.
 * @param[in]  width, height  image size.
 *
 */

void divergence(float ***p, float **div_p, int num_channels, int width,
                int height)
{
	int num_pixels = width*height;
	
	for (int c = 0; c<  num_channels; c++)
	{
		// Filling div_p with zeros
		fpClear(div_p[c], 0.0f, num_pixels);	

		// Compute divergence as <-div p, u> = <p, grad u>
		for (int j = 0; j < height; j++) 
		{			
			int k = j * width;

			for (int i = 0; i < width; i++) 
			{
				if(i != width-1) 
				{
					//grad[c][0][i+k] = data[c][i+1+k]-data[c][i+k];
					div_p[c][i+1+k] -= p[c][0][i+k]; 
					div_p[c][i+k] += p[c][0][i+k];
				}

				if(j != height-1)
				{ 
					//grad[c][1][i+k] = data[c][i+width+k]-data[c][i+k];
					div_p[c][i+width+k] -= p[c][1][i+k];
					div_p[c][i+k] += p[c][1][i+k];
				}			
			}		
		}
	}	
}

/**
 * \brief  Compute RMSE of two images.
 *
 * @param[in] u, v  input images : the first pointer accounts for the channel
 *            and the second one for the pixel position.
 * @param[in] num_channels  number of channels of both images.
 * @param[in] num_pixels  number of pixels of both images.
 * @return RMSE of @f$ u-v @f$:
 *
 * @f$ RMSE = \sqrt{\frac{\sum_{c=1}^{num\_channels} \sum_{i=1}^{num\_pixels}
 * (u[c][i] - v[c][i])^2}{num\_channels * num\_pixels}} @f$.
 *
 */

float compute_RMSE(float **u, float **v, int num_channels, int num_pixels)
{
    int N = num_pixels * num_channels;
	float RMSE = 0.0f;

	for(int c = 0; c < num_channels; c++) 
	    for(int i = 0; i < num_pixels; i++) 
		{
			float value = u[c][i] - v[c][i];
			RMSE += value * value;
		}

	return sqrt(RMSE / N);
}

/**
 * \brief  Compute maximum norm between two iterations of the dual variable.
 *
 * @param[in] p1, p2  dual variables : the first pointer accounts for the
 *            channel, the second one for the coordinate (x or y) and the
 *            third one for the pixel position.
 * @param[in] num_channels  number of channels of images.
 * @param[in] num_pixels  number of pixels of images.
 * @return maximum variation of @f$ p_1 - p_2 @f$.
 *
 * @f$ max\_norm = \max_{c,j,i}\{ p_1[c][j][i] - p_2[c][j][i] \} @f$.
 */

float max_norm(float ***p1, float ***p2, int num_channels, int num_pixels)
{
	float dif = 0.0f;

    for(int c = 0; c < num_channels; c++)
        for(int j = 0; j < 2; j++) 
            for(int i = 0; i < num_pixels; i++) 
                dif = MAX (fabs(p1[c][j][i] - p2[c][j][i]), dif);	
		 
	return dif;
}

/**
 * \brief Chambolle's projection algorithm for TV-denoising.
 *
 * The input image is processed as follows :
 *
 * @li @f$ p^0 = (0,0) @f$;
 * @li Repeat for each channel
 * @f$ p_{i,j}^{k+1} = \frac{p_{i,j}^k + h_t \left( \nabla \left(
 *     \mbox{div} p^k - \lambda f \right) \right)_{ij}} {1 + h_t \big|
 *     \left( \nabla \left( \mbox{div} p^k -\lambda f \right)
 *     \right)_{ij}\big|} @f$
 * until convergence.
 * @li Denoised image is given by
 * @f$ u = f - \frac{1}{\lambda}\mbox{div} p @f$
 *
 * @param[out]  u  denoised image : the first pointer accounts for the number
 *              of channels and the second one for the pixel position.
 * @param[in]   f  noisy image : the first pointer accounts for the number of
 *              channels and the second one for the pixel position.
 * @param[in]   p  initial dual variable : the first pointer accounts for the
 *              number of channels, the second one for the coordinate (x or y)
 *              and the third one for the pixel position.
 * @param[out]  p  final dual variable : the first pointer accounts for the
 *              number of channels, the second one for the coordinate (x or y)
 *              and the third one for the pixel position.
 * @param[in]   lambda  trade-off parameter.
 * @param[in]   tolerance  algorithm tolerance.
 * @param[in]   num_channels  number of channels of the image.
 * @param[in]   width, height  image size.
 *
 */

void chambolle(float **u, float **f, float ***p, float lambda, float tolerance,
               int num_channels, int width, int height)
{
    int num_pixels = width * height;

	// Create auxiliar vectors
	float ***grad = new float**[num_channels];
	float ***p_updated = new float**[num_channels];
	float **div_p = new float*[num_channels];
	float **v = new float*[num_channels];

	for(int c = 0; c < num_channels; c++) 
	{	
		grad[c] = new float*[2];
		p_updated[c] = new float*[2];
		
		for(int j = 0; j < 2; j++)
		{
			grad[c][j] = new float[num_pixels];
			p_updated[c][j] = new float[num_pixels];
		}
		
		div_p[c] = new float[num_pixels];
		v[c] = new float[num_pixels];
	}	

	// Compute divergence of initial p
	divergence(p, div_p, num_channels, width, height);	

	// Iterative process
	float dt = 0.248f;
	float dif = tolerance;

	while(dif >= tolerance)
	{
		// Compute argument of gradient
		for(int c = 0; c < num_channels; c++) 
			for(int i = 0; i < num_pixels; i++)
				v[c][i] = div_p[c][i] - lambda * f[c][i];

		// Compute gradient
		gradient(v, grad, num_channels, width, height);

		// Update dual variable using projection algorithm
		for(int i = 0; i < num_pixels; i++)
		{
			float normgrad = 0.0f;

			// Compute euclidean norm of the gradient
			for(int c = 0; c < num_channels; c++) 
			{
				float ux = grad[c][0][i];
				float uy = grad[c][1][i];
				normgrad += ux * ux + uy * uy;
			}

			normgrad = sqrt(normgrad);

			// Apply numerical scheme
			float denom = 1.0f + dt * normgrad;

			for(int c = 0; c < num_channels; c++)
				for(int j = 0; j < 2; j++)
					p_updated[c][j][i] = (p[c][j][i] + dt * grad[c][j][i]) /
                                         denom;
		}

		// Compute maximum variation between two consecutive iterations
		dif = max_norm(p, p_updated, num_channels, num_pixels);		

		// Update dual variable
		for(int c = 0; c < num_channels; c++) 
			for(int j = 0; j < 2; j++)
				fpCopy(p_updated[c][j], p[c][j], num_pixels);

		// Compute divergence of the dual variable
		divergence(p, div_p, num_channels, width, height);
	}

	// Compute Chambolle's solution
	float divlambda = 1.0f / lambda;

	for(int c = 0; c < num_channels; c++)
		for(int i = 0; i < num_pixels; i++)
			u[c][i] = f[c][i] - divlambda * div_p[c][i];

	// Free memory
	for(int c = 0; c < num_channels; c++) 
	{		
		for(int j = 0; j < 2; j++)
        {		
			delete[] grad[c][j];
            delete[] p_updated[c][j];
		}
        
		delete[] grad[c];
        delete[] p_updated[c];
        delete[] div_p[c];
        delete[] v[c];
	}
    
	delete[] div_p;
    delete[] v;
    delete[] grad;
    delete[] p_updated;
}

/**
 * \brief  Tunes lambda parameter.   	
 *
 * @param[out]  u  denoised image : the first pointer accounts for the number
 *              of channels and the second one for the pixel position.
 * @param[in]   f  noisy image : the first pointer accounts for the number of
 *              channels and the second one for the pixel position.
 * @param[in]   p  initial dual variable : the first pointer accounts for the
 *              number of channels, the second one for the coordinate (x or y)
 *              and the third one for the pixel position.
 * @param[out]  p  final dual variable : the first pointer accounts for the
 *              number of channels, the second one for the coordinate (x or y)
 *              and the third one for the pixel position.
 * @param[in]   sigma  noise standard deviation.
 * @param[in]   lambdaIter  number of iterations for lambda tunning.
 * @param[in]   lambdaTol  Chambolle's algorithm tolerance.
 * @param[in]   num_channels  number of channels of the image.
 * @param[in]   width, height  image size.
 * @return final lambda value.
 *
 */

float lambdaTuning(float **u, float **f, float ***p, float sigma,
                    int lambdaIter, float lambdaTol, int num_channels,
                    int width, int height)
{
    int num_pixels = width * height;

	// Initialize lambda
    float lambda = 2.1237f / sigma + 2.0547f / (sigma * sigma);
    lambda /= num_channels;
    
	if(lambda < 0.001f)  //Prevent too small lambda
		lambda = 0.001f;

    printf(":: Tuning lambda ::\n");
	printf("%e\n", lambda);
	
	// Tuning lambda. Each chambolle uses the current p as the initial guess
    // and overwrites it. This speeds up the computation because the result
    // using the previous lambda value is a good estimate for next lambda value
	for(int k = 0; k < lambdaIter; k++)
	{
		chambolle(u, f, p, lambda, lambdaTol, num_channels, width, height);

		float RMSE = compute_RMSE(u, f, num_channels, num_pixels);
		lambda *= RMSE / sigma;

		if (lambda < 0.001f)  //Prevent too small lambda
			lambda = 0.001f;

		printf("%e\n", lambda);
	}

	return lambda;
}

/**
 * \brief  Perform TV-denoising.   	
 *
 * @param[in]   option  option algorithm selected.
 * @param[out]  u  denoised image : the first pointer accounts for the number
 *              of channels and the second one for the pixel position.
 * @param[in]   f  noisy image : the first pointer accounts for the number of
 *              channels and the second one for the pixel position.
 * @param[in]   sigma  noise standard deviation.
 * @param[in]   lambda_fixed  fixed trade-off parameter if desired.
 * @param[in]   num_channels  number of channels of the image.
 * @param[in]   width, height  image size.
 * @return 1 if exit success.
 *
 */

int TVdenoising(int option, float **u, float **f, float sigma,
                float lambda_fixed, int num_channels, int width, int height)
{
    int num_pixels = width * height;
    
	// Dual variable
	float ***p = new float**[num_channels];
	for(int c = 0; c < num_channels; c++) 
	{
		p[c] = new float*[2];
		for (int j = 0; j < 2; j++) 
		{
			p[c][j] = new float[num_pixels];	
			fpClear(p[c][j], 0.0f, num_pixels);		
		}
	}

	// Tunning lambda or fixed lambda
    float lambda;

    if(option == 1 || option == 3)
    {
	    int lambdaIter = 5;
	    float lambdaTol = 0.01f;
	    lambda = lambdaTuning(u, f, p, sigma, lambdaIter, lambdaTol,
                              num_channels, width, height);
    } else
        lambda = lambda_fixed;

	// Final denoising
	float tolerance = 0.001f;
	// float tolerance = 0.0001f;
    
	for(int c = 0; c < num_channels; c++)
		for (int j = 0; j < 2; j++)
			fpClear(p[c][j], 0.0f, num_pixels);
    
	chambolle(u, f, p, lambda, tolerance, num_channels, width, height);

	// Free memory
	for(int c = 0; c < num_channels; c++)
    {
		for(int j = 0; j < 2; j++) 	
			delete[] p[c][j];
	
        delete[] p[c];
    }
    
    delete[] p;

	return 1;
}