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

struct DATA_CSPMV{
    float A[DATASIZE * DATASIZE * 2];
    float a[DATASIZE * (DATASIZE + 1)];
    float b[DATASIZE * 2 * INCREMENT];
    float c[DATASIZE * 2 * INCREMENT];
    float C[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX
static struct DATA_CSPMV data_cspmv;

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
* output param data_cspmv.C - matrix computed by gemv
*/
static void cspmv_trusted(char uplo, blasint n, float *alpha, float *a,
                          float *b, blasint inc_b, float *beta, float *c,
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
                    data_cspmv.A[i * n * 2 + j] = 
                            data_cspmv.A[j * n + i * 2];
                    data_cspmv.A[i * n * 2 + j + 1] = 
                            data_cspmv.A[j * n + i * 2 + 1];
                }
                else
                {
                    data_cspmv.A[i * n * 2 + j] = a[k++];
                    data_cspmv.A[i * n * 2 + j + 1] = a[k++];
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
                    data_cspmv.A[i * n * 2 + j] = 
                            data_cspmv.A[(j - 1) * n + i * 2 + 1];
                    data_cspmv.A[i * n * 2 + j - 1] = 
                            data_cspmv.A[(j - 1) * n + i * 2];
                }
                else
                {
                    data_cspmv.A[i * n * 2 + j] = a[k--];
                    data_cspmv.A[i * n * 2 + j - 1] = a[k--];
                }
            }
        }
    }

    // Run gemv with the unpacked matrix
    BLASFUNC(cgemv)(&trans, &n, &n, alpha, data_cspmv.A, &n, b, 
                    &inc_b, beta, data_cspmv.C, &inc_c);
}

static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
* Comapare results computed by cspmv and cspmv_trusted
*
* param uplo specifies whether matrix A is upper or lower triangular
* param n - number of rows and columns of A
* param alpha - scaling factor for the matrix-vector product
* param inc_b - stride of vector b
* param beta - scaling factor for vector c
* param inc_c - stride of vector c
* return norm of differences
*/
static float check_cspmv(char uplo, blasint n, float *alpha, blasint inc_b,
                          float *beta, blasint inc_c)
{
    blasint i;

    // Fill symmetric packed maxtix a, vectors b and c 
    rand_generate(data_cspmv.a, n * (n + 1));
    rand_generate(data_cspmv.b, 2 * n * inc_b);
    rand_generate(data_cspmv.c, 2 * n * inc_c);

    // Copy vector c for cspmv_trusted
    for (i = 0; i < n * 2 * inc_c; i++)
        data_cspmv.C[i] = data_cspmv.c[i];

    cspmv_trusted(uplo, n, alpha, data_cspmv.a, data_cspmv.b, 
                  inc_b, beta, data_cspmv.C, inc_c);
    BLASFUNC(cspmv)(&uplo, &n, alpha, data_cspmv.a, data_cspmv.b, 
                    &inc_b, beta, data_cspmv.c, &inc_c);

    // Find the differences between output vector computed by cspmv and cspmv_trusted
    for (i = 0; i < n * 2 * inc_c; i++)
        data_cspmv.c[i] -= data_cspmv.C[i];

    // Find the norm of differences
    return BLASFUNC(scnrm2)(&n, data_cspmv.c, &inc_c);
}

/**
* Check if error function was called with expected function name
* and param info
*
* param uplo specifies whether matrix A is upper or lower triangular
* param n - number of rows and columns of A
* param inc_b - stride of vector b
* param inc_c - stride of vector c
* param expected_info - expected invalid parameter number in cspmv
* return TRUE if everything is ok, otherwise FALSE
*/
static int check_badargs(char uplo, blasint n, blasint inc_b,
                          blasint inc_c, int expected_info)
{
    float alpha[] = {1.0, 1.0};
    float beta[] = {0.0, 0.0};

    set_xerbla("CSPMV ", expected_info);

    BLASFUNC(cspmv)(&uplo, &n, alpha, data_cspmv.a, data_cspmv.b, 
                    &inc_b, beta, data_cspmv.c, &inc_c);

    return check_error();
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 1
*/
CTEST(cspmv, upper_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 2
*/
CTEST(cspmv, upper_inc_b_1_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 2;
    char uplo = 'U';
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 1
*/
CTEST(cspmv, upper_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    char uplo = 'U';
    float alpha[] = {1.0f, 0.0f};
    float beta[] = {1.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is upper triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 2
*/
CTEST(cspmv, upper_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    char uplo = 'U';
    float alpha[] = {2.5, -2.1};
    float beta[] = {0.0f, 1.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 1
*/
CTEST(cspmv, lower_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'L';
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 1
* Stride of vector c is 2
*/
CTEST(cspmv, lower_inc_b_1_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 2;
    char uplo = 'L';
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 1
*/
CTEST(cspmv, lower_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    char uplo = 'L';
    float alpha[] = {1.0f, 0.0f};
    float beta[] = {1.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Test cspmv by comparing it against cgemv
* with the following options:
*
* A is lower triangular
* Number of rows and columns of A is 100
* Stride of vector b is 2
* Stride of vector c is 2
*/
CTEST(cspmv, lower_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    char uplo = 'L';
    float alpha[] = {2.5, -2.1};
    float beta[] = {0.0f, 1.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
* Check if output matrix A contains any NaNs
*/
CTEST(cspmv, check_for_NaN)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_cspmv(uplo, N, alpha, inc_b, beta, inc_c);

    ASSERT_TRUE(norm == norm); /* NaN == NaN is false */
}

/**
* Test error function for an invalid param uplo.
* uplo specifies whether A is upper or lower triangular.
*/
CTEST(cspmv, xerbla_uplo_invalid)
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
CTEST(cspmv, xerbla_N_invalid)
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
CTEST(cspmv, xerbla_inc_b_zero)
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
CTEST(cspmv, xerbla_inc_c_zero)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 0;
    char uplo = 'U';
    int expected_info = 9;
    int passed;

    passed = check_badargs(uplo, N, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif
