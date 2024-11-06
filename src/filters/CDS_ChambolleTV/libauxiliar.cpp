/*
 * Copyright 2009-201 IPOL Image Processing On Line http://www.ipol.im/
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
 * @file libauxiliar.cpp
 * @brief auxiliar functions.
 * @author Joan Duran <joan.duran@uib.es>
 */

#include "libauxiliar.h"

/**
 * \brief  Initialize a float vector. 
 *
 * @param[in]  u  vector input.
 * @param[out] u  vector output.
 * @param[in]  value  value inserted.
 * @param[in]  dim  size of the vector.
 *
 */

void fpClear(float *u, float value, int dim) 
{
    for(int i = 0; i < dim; i++) 
	    u[i] = value;
}

/**
 * \brief  Copy the values of a float vector into another.
 *
 * @param[in]  input  vector input.
 * @param[out] output  vector output.
 * @param[in]  dim  size of vectors.
 *
 */

void fpCopy(float *input, float *output, int dim)
{
    if (input != output)  
        memcpy((void *) output, (const void *) input, dim * sizeof(float));
}

/**
 * \brief  Add white Gaussian noise to an image.
 *
 * @param[in]  u  original image.
 * @param[out] v  noised image.
 * @param[in]  std  noise standard deviation.
 * @param[in]  randinit  random parameter.
 * @param[in]  dim  image size.
 *
 */

void fiAddNoise(float *u, float *v, float std, long int randinit, int dim)
{
    mt_init_genrand((unsigned long int) time (NULL) + 
                    (unsigned long int) getpid() + 
                    (unsigned long int) randinit);

    for(int i = 0; i < dim; i++)
    {
        double a = mt_genrand_res53();
        double b = mt_genrand_res53();
        double z = (double)(std) * sqrt(-2.0 * log(a)) * cos(2.0 * M_PI * b);

        v[i] =  u[i] + (float) z;
    }
}
