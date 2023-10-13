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

struct DATA_ZSCAL {
    double x_test[DATASIZE * 2 * INCREMENT];
    double x_verify[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX16
static struct DATA_ZSCAL data_zscal;


/**
 * zscal reference code
 *
 * param n - number of elements of vector x
 * param alpha - scaling factor for the vector product
 * param x - buffer holding input vector x
 * param inc - stride of vector x
 */
static void zscal_trusted(blasint n, double *alpha, double* x, blasint inc){
    blasint i, ip = 0;
    blasint inc_x2 = 2 * inc;
    double temp;
    for (i = 0; i < n; i++)
	{
        temp = alpha[0] * x[ip] - alpha[1] * x[ip+1];
		x[ip+1] = alpha[0] * x[ip+1] + alpha[1] * x[ip];
        x[ip] = temp;
        ip += inc_x2;
    }
}

/**
 * Comapare results computed by zscal and zscal_trusted
 *
 * param api specifies tested api (C or Fortran)
 * param n - number of elements of vector x
 * param alpha - scaling factor for the vector product
 * param inc - stride of vector x
 * return norm of differences
 */
static double check_zscal(char api, blasint n, double *alpha, blasint inc)
{
    blasint i;

    // Fill vectors x
    drand_generate(data_zscal.x_test, n * inc * 2);

    // Copy vector x for zscal_trusted
    for (i = 0; i < n * 2 * inc; i++)
        data_zscal.x_verify[i] = data_zscal.x_test[i];

    zscal_trusted(n, alpha, data_zscal.x_verify, inc);

    if(api == 'F')
        BLASFUNC(zscal)(&n, alpha, data_zscal.x_test, &inc);
    else
        cblas_zscal(n, alpha, data_zscal.x_test, inc);

    // Find the differences between output vector computed by zscal and zscal_trusted
    for (i = 0; i < n * 2 * inc; i++)
        data_zscal.x_verify[i] -= data_zscal.x_test[i];

    // Find the norm of differences
    return BLASFUNC(dznrm2)(&n, data_zscal.x_verify, &inc);
}

/**
 * Fortran API specific test
 * Test zscal by comparing it against reference
 */
CTEST(zscal, alpha_r_zero_alpha_i_not_zero)
{
    blasint N = DATASIZE;
    blasint inc = 1;
    double alpha[2] = {0.0, 1.0};

    double norm = check_zscal('F', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zscal by comparing it against reference
 */
CTEST(zscal, alpha_r_zero_alpha_i_zero_inc_2)
{
    blasint N = DATASIZE;
    blasint inc = 2;
    double alpha[2] = {0.0, 0.0};

    double norm = check_zscal('F', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zscal by comparing it against reference
 */
CTEST(zscal, c_api_alpha_r_zero_alpha_i_not_zero)
{
    blasint N = DATASIZE;
    blasint inc = 1;
    double alpha[2] = {0.0, 1.0};

    double norm = check_zscal('C', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zscal by comparing it against reference
 */
CTEST(zscal, c_api_alpha_r_zero_alpha_i_zero_inc_2)
{
    blasint N = DATASIZE;
    blasint inc = 2;
    double alpha[2] = {0.0, 0.0};

    double norm = check_zscal('C', N, alpha, inc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}
#endif