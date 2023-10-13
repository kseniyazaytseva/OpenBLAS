/*
 * Copyright (C), 2023, KNS Group LLC (YADRO).
 * All Rights Reserved.
 *
 * This software contains the intellectual property of YADRO
 * or is licensed to YADRO from third parties. Use of this
 * software and the intellectual property contained therein is expressly
 * limited to the terms and conditions of the License Agreement under which
 * it is provided by YADRO.
 */

#include "utest/openblas_utest.h"
#include <cblas.h>
#include "common.h"

#define DATASIZE 100
#define INCREMENT 2

struct DATA_CSCAL {
    float x_test[DATASIZE * 2 * INCREMENT];
    float x_verify[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX
static struct DATA_CSCAL data_cscal;

/**
 * cscal reference code
 *
 * param n - number of elements of vector x
 * param alpha - scaling factor for the vector product
 * param x - buffer holding input vector x
 * param inc - stride of vector x
 */
static void cscal_trusted(blasint n, float *alpha, float* x, blasint inc){
    blasint i, ip = 0;
    blasint inc_x2 = 2 * inc;
    float temp;
    for (i = 0; i < n; i++)
	{
        temp = alpha[0] * x[ip] - alpha[1] * x[ip+1];
		x[ip+1] = alpha[0] * x[ip+1] + alpha[1] * x[ip];
        x[ip] = temp;
        ip += inc_x2;
    }
}

/**
 * Comapare results computed by cscal and cscal_trusted
 *
 * param api specifies tested api (C or Fortran)
 * param n - number of elements of vector x
 * param alpha - scaling factor for the vector product
 * param inc - stride of vector x
 * return norm of differences
 */
static float check_cscal(char api, blasint n, float *alpha, blasint inc)
{
    blasint i;

    // Fill vectors a 
    srand_generate(data_cscal.x_test, n * inc * 2);

    // Copy vector x for cscal_trusted
    for (i = 0; i < n * 2 * inc; i++)
        data_cscal.x_verify[i] = data_cscal.x_test[i];

    cscal_trusted(n, alpha, data_cscal.x_verify, inc);

    if(api == 'F')
        BLASFUNC(cscal)(&n, alpha, data_cscal.x_test, &inc);
    else
        cblas_cscal(n, alpha, data_cscal.x_test, inc);

    // Find the differences between output vector computed by cscal and cscal_trusted
    for (i = 0; i < n * 2 * inc; i++)
        data_cscal.x_verify[i] -= data_cscal.x_test[i];

    // Find the norm of differences
    return BLASFUNC(scnrm2)(&n, data_cscal.x_verify, &inc);
}

/**
 * Fortran API specific test
 * Test cscal by comparing it against reference
 */
CTEST(cscal, alpha_r_zero_alpha_i_not_zero)
{
    blasint N = DATASIZE;
    blasint inc = 1;
    float alpha[2] = {0.0f, 1.0f};

    float norm = check_cscal('F', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test cscal by comparing it against reference
 */
CTEST(cscal, alpha_r_zero_alpha_i_zero_inc_2)
{
    blasint N = DATASIZE;
    blasint inc = 2;
    float alpha[2] = {0.0f, 0.0f};

    float norm = check_cscal('F', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test cscal by comparing it against reference
 */
CTEST(cscal, c_api_alpha_r_zero_alpha_i_not_zero)
{
    blasint N = DATASIZE;
    blasint inc = 1;
    float alpha[2] = {0.0f, 1.0f};

    float norm = check_cscal('C', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test cscal by comparing it against reference
 */
CTEST(cscal, c_api_alpha_r_zero_alpha_i_zero_inc_2)
{
    blasint N = DATASIZE;
    blasint inc = 2;
    float alpha[2] = {0.0f, 0.0f};

    float norm = check_cscal('C', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif