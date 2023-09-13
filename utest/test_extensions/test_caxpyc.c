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

#define DATASIZE 100
#define INCREMENT 2

struct DATA_СAXPYC
{
    float x_test[DATASIZE * INCREMENT * 2];
    float x_verify[DATASIZE * INCREMENT * 2];
    float y_test[DATASIZE * INCREMENT * 2];
    float y_verify[DATASIZE * INCREMENT * 2];
};
#ifdef BUILD_COMPLEX
static struct DATA_СAXPYC data_caxpyc;

/**
 * Generate random vector stored in one-dimensional array
 */
static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
 * Conjugate vector stored in one-dimensional array
 */
static void conjugate(blasint n, float *a, blasint inc)
{
    blasint i;
    for (i = 1; i < n * 2 * inc; i += 2 * inc)
        a[i] *= -1.0f;
}

/**
 * Test caxpyc by conjugating vector x and comparing with caxpy.
 * Compare with the following options:
 *
 * param n - number of elements in vectors x and y
 * param alpha - scalar alpha
 * param incx - increment for the elements of x
 * param incy - increment for the elements of y
 * return norm of difference
 */
static float check_caxpyc(blasint n, float *alpha, blasint incx, blasint incy)
{
    blasint i;

    rand_generate(data_caxpyc.x_test, n * incx * 2);
    rand_generate(data_caxpyc.y_test, n * incy * 2);

    for (i = 0; i < n * incx * 2; i++)
        data_caxpyc.x_verify[i] = data_caxpyc.x_test[i];

    for (i = 0; i < n * incy * 2; i++)
        data_caxpyc.y_verify[i] = data_caxpyc.y_test[i];

    conjugate(n, data_caxpyc.x_verify, incx);

    BLASFUNC(caxpy)
    (&n, alpha, data_caxpyc.x_verify, &incx,
     data_caxpyc.y_verify, &incy);

    BLASFUNC(caxpyc)
    (&n, alpha, data_caxpyc.x_test, &incx,
     data_caxpyc.y_test, &incy);

    for (i = 0; i < n * incy * 2; i++)
        data_caxpyc.y_verify[i] -= data_caxpyc.y_test[i];

    return BLASFUNC(scnrm2)(&n, data_caxpyc.y_verify, &incy);
}

/**
 * Test caxpyc by conjugating vector x and comparing with caxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 */
CTEST(caxpyc, conj_strides_one)
{
    blasint n = DATASIZE, incx = 1, incy = 1;
    float alpha[] = {5.0f, 2.2f};

    float norm = check_caxpyc(n, alpha, incx, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test caxpyc by conjugating vector x and comparing with caxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 */
CTEST(caxpyc, conj_incx_one)
{
    blasint n = DATASIZE, incx = 1, incy = 2;
    float alpha[] = {5.0f, 2.2f};

    float norm = check_caxpyc(n, alpha, incx, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test caxpyc by conjugating vector x and comparing with caxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 */
CTEST(caxpyc, conj_incy_one)
{
    blasint n = DATASIZE, incx = 2, incy = 1;
    float alpha[] = {5.0f, 2.2f};

    float norm = check_caxpyc(n, alpha, incx, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test caxpyc by conjugating vector x and comparing with caxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 */
CTEST(caxpyc, conj_strides_two)
{
    blasint n = DATASIZE, incx = 2, incy = 2;
    float alpha[] = {5.0f, 2.2f};

    float norm = check_caxpyc(n, alpha, incx, incy);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif
