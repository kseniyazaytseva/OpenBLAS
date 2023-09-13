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

struct DATA_ZAXPYC
{
	double x_test[DATASIZE * INCREMENT * 2];
	double x_verify[DATASIZE * INCREMENT * 2];
	double y_test[DATASIZE * INCREMENT * 2];
	double y_verify[DATASIZE * INCREMENT * 2];
};
#ifdef BUILD_COMPLEX16
static struct DATA_ZAXPYC data_zaxpyc;

/**
 * Generate random vector stored in one-dimensional array
 */
static void rand_generate(double *a, blasint n)
{
	blasint i;
	for (i = 0; i < n; i++)
		a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * Conjugate vector stored in one-dimensional array
 */
static void conjugate(blasint n, double *a, blasint inc)
{
	blasint i;
	for (i = 1; i < n * 2 * inc; i += 2 * inc)
		a[i] *= -1.0;
}

/**
 * Test zaxpyc by conjugating vector x and comparing with zaxpy.
 * Compare with the following options:
 *
 * param n - number of elements in vectors x and y
 * param alpha - scalar alpha
 * param incx - increment for the elements of x
 * param incy - increment for the elements of y
 * return norm of difference
 */
static double check_zaxpyc(blasint n, double *alpha, blasint incx, blasint incy)
{
	blasint i;

	rand_generate(data_zaxpyc.x_test, n * incx * 2);
	rand_generate(data_zaxpyc.y_test, n * incy * 2);

	for (i = 0; i < n * incx * 2; i++)
		data_zaxpyc.x_verify[i] = data_zaxpyc.x_test[i];

	for (i = 0; i < n * incy * 2; i++)
		data_zaxpyc.y_verify[i] = data_zaxpyc.y_test[i];

	conjugate(n, data_zaxpyc.x_verify, incx);

	BLASFUNC(zaxpy)
	(&n, alpha, data_zaxpyc.x_verify, &incx,
	 data_zaxpyc.y_verify, &incy);

	BLASFUNC(zaxpyc)
	(&n, alpha, data_zaxpyc.x_test, &incx,
	 data_zaxpyc.y_test, &incy);

	for (i = 0; i < n * incy * 2; i++)
		data_zaxpyc.y_verify[i] -= data_zaxpyc.y_test[i];

	return BLASFUNC(dznrm2)(&n, data_zaxpyc.y_verify, &incy);
}

/**
 * Test zaxpyc by conjugating vector x and comparing with zaxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 */
CTEST(zaxpyc, conj_strides_one)
{
	blasint n = DATASIZE, incx = 1, incy = 1;
	double alpha[] = {5.0, 2.2};

	double norm = check_zaxpyc(n, alpha, incx, incy);

	ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zaxpyc by conjugating vector x and comparing with zaxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 */
CTEST(zaxpyc, conj_incx_one)
{
	blasint n = DATASIZE, incx = 1, incy = 2;
	double alpha[] = {5.0, 2.2};

	double norm = check_zaxpyc(n, alpha, incx, incy);

	ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zaxpyc by conjugating vector x and comparing with zaxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 */
CTEST(zaxpyc, conj_incy_one)
{
	blasint n = DATASIZE, incx = 2, incy = 1;
	double alpha[] = {5.0, 2.2};

	double norm = check_zaxpyc(n, alpha, incx, incy);

	ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zaxpyc by conjugating vector x and comparing with zaxpy.
 * Test with the following options:
 *
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 */
CTEST(zaxpyc, conj_strides_two)
{
	blasint n = DATASIZE, incx = 2, incy = 2;
	double alpha[] = {5.0, 2.2};

	double norm = check_zaxpyc(n, alpha, incx, incy);

	ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}
#endif
