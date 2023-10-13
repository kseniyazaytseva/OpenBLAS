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
#include "common.h"

#define DATASIZE 100
#define INCREMENT 2

struct DATA_CAXPBY {
    float x_test[DATASIZE * INCREMENT * 2];
    float x_verify[DATASIZE * INCREMENT * 2];
    float y_test[DATASIZE * INCREMENT * 2];
    float y_verify[DATASIZE * INCREMENT * 2];
};

#ifdef BUILD_COMPLEX
static struct DATA_CAXPBY data_caxpby;

/**
 * Fortran API specific function
 * Test caxpby by comparing it with cscal and caxpy.
 * Compare with the following options:
 * 
 * param n - number of elements in vectors x and y
 * param alpha - scalar alpha
 * param incx - increment for the elements of x
 * param beta - scalar beta
 * param incy - increment for the elements of y
 * return norm of difference
 */
static float check_caxpby(blasint n, float *alpha, blasint incx, float *beta, blasint incy)
{
    blasint i;

    // cscal accept only positive increments
    blasint incx_abs = labs(incx);
    blasint incy_abs = labs(incy);

    // Fill vectors x, y
    srand_generate(data_caxpby.x_test, n * incx_abs * 2);
    srand_generate(data_caxpby.y_test, n * incy_abs * 2);

    // Copy vector x for caxpy
    for (i = 0; i < n * incx_abs * 2; i++)
        data_caxpby.x_verify[i] = data_caxpby.x_test[i];

    // Copy vector y for cscal
    for (i = 0; i < n * incy_abs * 2; i++)
        data_caxpby.y_verify[i] = data_caxpby.y_test[i];

    // Find beta*y
    BLASFUNC(cscal)(&n, beta, data_caxpby.y_verify, &incy_abs);

    // Find sum of alpha*x and beta*y
    BLASFUNC(caxpy)(&n, alpha, data_caxpby.x_verify, &incx,
                        data_caxpby.y_verify, &incy);
    
    BLASFUNC(caxpby)(&n, alpha, data_caxpby.x_test, &incx,
                        beta, data_caxpby.y_test, &incy);

    // Find the differences between output vector caculated by caxpby and caxpy
    for (i = 0; i < n * incy_abs * 2; i++)
        data_caxpby.y_test[i] -= data_caxpby.y_verify[i];

    // Find the norm of differences
    return BLASFUNC(scnrm2)(&n, data_caxpby.y_test, &incy_abs);
}

/**
 * C API specific function 
 * Test caxpby by comparing it with cscal and caxpy.
 * Compare with the following options:
 * 
 * param n - number of elements in vectors x and y
 * param alpha - scalar alpha
 * param incx - increment for the elements of x
 * param beta - scalar beta
 * param incy - increment for the elements of y
 * return norm of difference
 */
static float c_api_check_caxpby(blasint n, float *alpha, blasint incx, float *beta, blasint incy)
{
    blasint i;

    // cscal accept only positive increments
    blasint incx_abs = labs(incx);
    blasint incy_abs = labs(incy);

    // Fill vectors x, y
    srand_generate(data_caxpby.x_test, n * incx_abs * 2);
    srand_generate(data_caxpby.y_test, n * incy_abs * 2);

    // Copy vector x for caxpy
    for (i = 0; i < n * incx_abs * 2; i++)
        data_caxpby.x_verify[i] = data_caxpby.x_test[i];

    // Copy vector y for cscal
    for (i = 0; i < n * incy_abs * 2; i++)
        data_caxpby.y_verify[i] = data_caxpby.y_test[i];

    // Find beta*y
    cblas_cscal(n, beta, data_caxpby.y_verify, incy_abs);

    // Find sum of alpha*x and beta*y
    cblas_caxpy(n, alpha, data_caxpby.x_verify, incx,
                        data_caxpby.y_verify, incy);
    
    cblas_caxpby(n, alpha, data_caxpby.x_test, incx,
                        beta, data_caxpby.y_test, incy);

    // Find the differences between output vector caculated by caxpby and caxpy
    for (i = 0; i < n * incy_abs * 2; i++)
        data_caxpby.y_test[i] -= data_caxpby.y_verify[i];

    // Find the norm of differences
    return cblas_scnrm2(n, data_caxpby.y_test, incy_abs);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 */
CTEST(caxpby, inc_x_1_inc_y_1_N_100)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 */
CTEST(caxpby, inc_x_2_inc_y_1_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = 1;
    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 */
CTEST(caxpby, inc_x_1_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = 1, incy = 2;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 */
CTEST(caxpby, inc_x_2_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = 2;
    float alpha[] = {3.0f, 1.0f};
    float beta[] = {4.0f, 3.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is 2
 */
CTEST(caxpby, inc_x_neg_1_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = -1, incy = 2;
    float alpha[] = {5.0f, 2.2f};
    float beta[] = {4.0f, 5.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is -1
 */
CTEST(caxpby, inc_x_2_inc_y_neg_1_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = -1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {6.0f, 3.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is -1
 */
CTEST(caxpby, inc_x_neg_2_inc_y_neg_1_N_100)
{
    blasint n = DATASIZE, incx = -2, incy = -1;
    float alpha[] = {7.0f, 2.0f};
    float beta[] = {3.5f, 1.3f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar alpha is zero
 */
CTEST(caxpby, inc_x_1_inc_y_1_N_100_alpha_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar beta is zero
 */
CTEST(caxpby, inc_x_1_inc_y_1_N_100_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar alpha is zero
 * Scalar beta is zero
 */
CTEST(caxpby, inc_x_1_inc_y_1_N_100_a_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * Scalar alpha is zero
 * Scalar beta is zero
*/
CTEST(caxpby, inc_x_1_inc_y_2_N_100_alpha_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 2;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Check if n - size of vectors x, y is zero
 */
CTEST(caxpby, check_n_zero)
{
    blasint n = 0, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = check_caxpby(n, alpha, incx, beta, incy);
    
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 */
CTEST(caxpby, c_api_inc_x_1_inc_y_1_N_100)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 */
CTEST(caxpby, c_api_inc_x_2_inc_y_1_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = 1;
    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 */
CTEST(caxpby, c_api_inc_x_1_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = 1, incy = 2;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {2.0f, 2.1f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 */
CTEST(caxpby, c_api_inc_x_2_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = 2;
    float alpha[] = {3.0f, 2.0f};
    float beta[] = {4.0f, 3.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is 2
 */
CTEST(caxpby, c_api_inc_x_neg_1_inc_y_2_N_100)
{
    blasint n = DATASIZE, incx = -1, incy = 2;
    float alpha[] = {5.0f, 2.0f};
    float beta[] = {4.0f, 3.1f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is -1
 */
CTEST(caxpby, c_api_inc_x_2_inc_y_neg_1_N_100)
{
    blasint n = DATASIZE, incx = 2, incy = -1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {6.0f, 2.3f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is -1
 */
CTEST(caxpby, c_api_inc_x_neg_2_inc_y_neg_1_N_100)
{
    blasint n = DATASIZE, incx = -2, incy = -1;
    float alpha[] = {7.0f, 1.0f};
    float beta[] = {3.5f, 1.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar alpha is zero
 */
CTEST(caxpby, c_api_inc_x_1_inc_y_1_N_100_alpha_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar beta is zero
 */
CTEST(caxpby, c_api_inc_x_1_inc_y_1_N_100_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * Scalar alpha is zero
 * Scalar beta is zero
 */
CTEST(caxpby, c_api_inc_x_1_inc_y_1_N_100_a_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test caxpby by comparing it with cscal and caxpy.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * Scalar alpha is zero
 * Scalar beta is zero
*/
CTEST(caxpby, c_api_inc_x_1_inc_y_2_N_100_alpha_beta_zero)
{
    blasint n = DATASIZE, incx = 1, incy = 2;
    float alpha[] = {0.0f, 0.0f};
    float beta[] = {0.0f, 0.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Check if n - size of vectors x, y is zero
 */
CTEST(caxpby, c_api_check_n_zero)
{
    blasint n = 0, incx = 1, incy = 1;
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    float norm = c_api_check_caxpby(n, alpha, incx, beta, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif
