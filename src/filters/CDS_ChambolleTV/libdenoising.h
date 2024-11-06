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

#ifndef _LIBDENOISING_H_
#define _LIBDENOISING_H_

#include "libauxiliar.h"

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
              int height);

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
                int height);

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

float compute_RMSE(float **u, float **v, int num_channels, int num_pixels);

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

float max_norm(float ***p1, float ***p2, int num_channels, int num_pixels);

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
               int num_channels, int width, int height);

/**
 * \brief  Tunnes lambda parameter.   	
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
                    int width, int height);

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
                float lambda_fixed, int num_channels, int width, int height);


void checkVarianceCondition(float **f, float sigma, int num_channels, int width,
                            int height);

#endif
