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

struct DATA_ZSPMV{
    double A[DATASIZE * DATASIZE * 2];
    double a[DATASIZE * (DATASIZE + 1)];
    double b[DATASIZE * 2 * INCREMENT];
    double c[DATASIZE * 2 * INCREMENT];
    double C[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX16
static struct DATA_ZSPMV data_zspmv;

/**
* Compute spmv via gemv since spmv is gemv for symmetric packed matrix
*
* param uplo specifies whether matrix A is upper or lower triangular
* param n - number of rows and columns of A
* param alpha - scaling factor for the matrix-vector product
* param a - buffer holding input matrix A
* param b - Buffer holding input vector b
* param inc_b - stride of vector b
* param beta - scaling factor for vector c
* param c - buffer holding input/output vector c
* param inc_c - stride of vector c
* output param data_zspmv.C - matrix computed by gemv
*/
static void zspmv_trusted(char uplo, blasint n, double *alpha, double *a,
                          double *b, blasint inc_b, double *beta, double *c,
                          blasint inc_c)
{
    blasint k;
    blasint i, j;

    // param for gemv (can use any, since the input matrix is symmetric)
    char trans = 'N';
    
    // Unpack the input symmetric packed matrix
    if (uplo == 'L')
    {
        k = 0;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n * 2; j += 2)
            {
                if (j / 2 < i)
                {
                    data_zspmv.A[i * n * 2 + j] = 
                            data_zspmv.A[j * n + i * 2];
                    data_zspmv.A[i * n * 2 + j + 1] = 
                            data_zspmv.A[j * n + i * 2 + 1];
                }
                else
                {
                    data_zspmv.A[i * n * 2 + j] = a[k++];
                    data_zspmv.A[i * n * 2 + j + 1] = a[k++];
                }
            }
        }
    }
    else
    {
        k = n * (n + 1) - 1;
        for (j = 2 * n - 1; j >= 0; j -= 2)
        {
            for (i = n - 1; i >= 0; i--)
            {
                if (j / 2 < i)
                {
                    data_zspmv.A[i * n * 2 + j] = 
                            data_zspmv.A[(j - 1) * n + i * 2 + 1];
                    data_zspmv.A[i * n * 2 + j - 1] = 
                            data_zspmv.A[(j - 1) * n + i * 2];
                }
                else
                {
                    data_zspmv.A[i * n * 2 + j] = a[k--];
                    data_zspmv.A[i * n * 2 + j - 1] = a[k--];
                }
            }
        }
    }

    // Run gemv with unpacked matrix
    BLASFUNC(zgemv)(&trans, &n, &n, alpha, data_zspmv.A, &n, b, 
                    &inc_b, beta, c, &inc_c);
}

static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
* Comapare results computed by zspmv and zspmv_trusted
*
* param uplo specifies whether matrix A is upper or lower triangular
* param n - number of rows and columns of A
* param alpha - scaling factor for the matrix-vector product
* param inc_b - stride of vector b
* param beta - scaling factor for vector c
* param inc_c - stride of vector c
* return norm of differences
*/
static double check_zspmv(char uplo, blasint n, double *alpha, blasint inc_b,
                          double *beta, blasint inc_c)
{
    blasint i;

    // Fill symmetric packed maxtix a, vectors b and c 
    rand_generate(data_zspmv.a, n * (n + 1));
    rand_generate(data_zspmv.b, 2 * n * inc_b);
    rand_generate(data_zspmv.c, 2 * n * inc_c);

    // Copy vector c for zspmv_trusted
    for (i = 0; i < n * 2 * inc_c; i++)
        data_zspmv.C[i] = data_zspmv.c[i];

    zspmv_trusted(uplo, n, alpha, data_zspmv.a, data_zspmv.b, 
                  inc_b, beta, data_zspmv.C, inc_c);
    BLASFUNC(zspmv)(&uplo, &n, alpha, data_zspmv.a, data_zspmv.b, 
                    &inc_b, beta, data_zspmv.c, &inc_c);

    // Find the differences between output vector caculated by zspmv and zspmv_trusted
    for (i = 0; i < n * 2 * inc_c; i++)
        data_zspmv.c[i] -= data_zspmv.C[i];

    // Find the norm of differences
    return BLASFUNC(dznrm2)(&n, data_zspmv.c, &inc_c);
}

/**
* Check if error function was called with expected function name
* and param info
*
* param uplo specifies whether matrix A is upper or lower triangular
* param n - number of rows and columns of A
* param inc_b - stride of vector b
* param inc_c - stride of vector c
* param expected_info - expected invalid parameter number in zspmv
* return TRUE if everything is ok, otherwise FALSE
*/
static int check_badargs(char uplo, blasint n, blasint inc_b,
                          blasint inc_c, int expected_info)
{
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};

    set_xerbla("ZSPMV ", expected_info);

    BLASFUNC(zspmv)(&uplo, &n, alpha, data_zspmv.a, data_zspmv.b, 
                    &inc_b, beta, data_zspmv.c, &inc_c);

    return check_error();
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 1
*/
CTEST(zspmv, upper_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 2
*/
CTEST(zspmv, upper_inc_b_1_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 2;
    char uplo = 'U';
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 1
*/
CTEST(zspmv, upper_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    char uplo = 'U';
    double alpha[] = {1.0, 0.0};
    double beta[] = {1.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 2
*/
CTEST(zspmv, upper_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    char uplo = 'U';
    double alpha[] = {2.5, -2.1};
    double beta[] = {0.0, 1.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 1
*/
CTEST(zspmv, lower_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'L';
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 2
*/
CTEST(zspmv, lower_inc_b_1_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 2;
    char uplo = 'L';
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 1
*/
CTEST(zspmv, lower_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    char uplo = 'L';
    double alpha[] = {1.0, 0.0};
    double beta[] = {1.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Test zspmv by comparing it against zgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 2
*/
CTEST(zspmv, lower_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    char uplo = 'L';
    double alpha[] = {2.5, -2.1};
    double beta[] = {0.0, 1.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
* Check if output matrix A contains any NaNs
*/
CTEST(zspmv, check_for_NaN)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    double alpha[] = {1.0, 1.0};
    double beta[] = {0.0, 0.0};
    double norm;

    norm = check_zspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_TRUE(norm == norm); /* NaN == NaN is false */
}

/**
* Test error function for an invalid param uplo.
* uplo specifies whether A is upper or lower triangular.
*/
CTEST(zspmv, xerbla_uplo_invalid)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'O';
    int expected_info = 1;
    int passed;

    passed = check_badargs(uplo, N, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param N -
* number of rows and columns of A. Must be at least zero.
*/
CTEST(zspmv, xerbla_N_invalid)
{
    blasint N = INVALID, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    int expected_info = 2;
    int passed;

    passed = check_badargs(uplo, N, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param inc_b -
* stride of vector b. Can't be zero.
*/
CTEST(zspmv, xerbla_inc_b_zero)
{
    blasint N = DATASIZE, inc_b = 0, inc_c = 1;
    char uplo = 'U';
    int expected_info = 6;
    int passed;

    passed = check_badargs(uplo, N, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param inc_c -
* stride of vector c. Can't be zero.
*/
CTEST(zspmv, xerbla_inc_c_zero)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 0;
    char uplo = 'U';
    int expected_info = 9;
    int passed;

    passed = check_badargs(uplo, N, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif
