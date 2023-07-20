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

struct DATA_csbmv{
    float tr_matrix[DATASIZE * (DATASIZE + 1)];
    float matrix[DATASIZE * DATASIZE * 2];
    float b[DATASIZE * 2 * INCREMENT];
    float c[DATASIZE * 2 * INCREMENT];
    float C[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX
static struct DATA_csbmv data_csbmv;
static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/** 
 * Transform full-storage symmetric band matrix A to upper (U) or lower (L)
 * band-packed storage mode.
 * 
 * param uplo specifies whether matrix a is upper or lower band-packed.
 * param n - number of rows and columns of A
 * param k - number of super-diagonals of A
 * output param a - buffer for holding symmetric band-packed matrix
 * param lda - specifies the leading dimension of a
 * param matrix - buffer holding full-storage symmetric band matrix A 
 * param ldm - specifies the leading dimension of A
*/
static void transform_to_band_storage(char uplo, blasint n, blasint k, float* a, blasint lda,
                                     float* matrix, blasint ldm) 
{
    blasint i, j, m;
    if (uplo == 'L') {
        for (j = 0; j < n; j++)
        {
            m = -j;
            for (i = 2 * j; i < MIN(2 * n, 2 * (j + k + 1)); i += 2)
            {
                a[(2*m + i) + j * lda * 2] = matrix[i + j * ldm * 2];
                a[(2*m + (i + 1)) + j * lda * 2] = matrix[(i + 1) + j * ldm * 2];
            }
        }
    }
    else {
        for (j = 0; j < n; j++)
        {   
            m = k - j;
            for (i = MAX(0, 2*(j - k)); i <= j*2; i += 2)
            {
                a[(2*m + i) + j * lda * 2] = matrix[i + j * ldm * 2];
                a[(2*m + (i + 1)) + j * lda * 2] = matrix[(i + 1) + j * ldm * 2];
            }
        }
    }
}

/** 
 * Generate full-storage symmetric band matrix A with k - super-diagonals
 * from input symmetric packed matrix in lower packed mode (L)
 * 
 * output param matrix - buffer for holding full-storage symmetric band matrix.
 * param tr_matrix - buffer holding input symmetric packed matrix
 * param n - number of rows and columns of A
 * param k - number of super-diagonals of A
*/
static void get_symmetric_band_matr(float *matrix, float *tr_matrix, blasint n, blasint k)
{
    blasint m;
    blasint i, j;
    m = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n * 2; j += 2)
        {
            // Make matrix band with k super-diagonals
            if (abs((i+1) - ceil((j+1)/2.0f)) > k) 
            {
                matrix[i * n * 2 + j] = 0.0f;
                matrix[i * n * 2 + j + 1] = 0.0f;
                continue;
            }

            if (j / 2 < i)
            {
                matrix[i * n * 2 + j] = 
                        matrix[j * n + i * 2];
                matrix[i * n * 2 + j + 1] = 
                        matrix[j * n + i * 2 + 1];
            }
            else
            {
                matrix[i * n * 2 + j] = tr_matrix[m++];
                matrix[i * n * 2 + j + 1] = tr_matrix[m++];
            }
        }
    }
}

/** 
 * Check if error function was called with expected function name
 * and param info
 * 
 * param uplo specifies whether matrix a is upper or lower band-packed.
 * param n - number of rows and columns of A
 * param k - number of super-diagonals of A
 * param lda - specifies the leading dimension of a
 * param inc_b - stride of vector b
 * param inc_c - stride of vector c
 * param expected_info - expected invalid parameter number in csbmv
 * return TRUE if everything is ok, otherwise FALSE 
*/
static int check_badargs(char uplo, blasint n, blasint k, blasint lda, blasint inc_b,
                          blasint inc_c, int expected_info)
{
    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};

    // Diagonal band matrix
    float a[2];
    rand_generate(a, 2);

    set_xerbla("CSBMV ", expected_info);

    BLASFUNC(csbmv)(&uplo, &n, &k, alpha, a, &lda, data_csbmv.b, 
                    &inc_b, beta, data_csbmv.c, &inc_c);

    return check_error();
}

/**
 * Comapare results computed by csbmv and cgemv 
 * since sbmv is gemv for symmetric band matrix
 * 
 * param uplo specifies whether matrix A is upper or lower triangular
 * param n - number of rows and columns of A
 * param k - number of super-diagonals of A
 * param alpha - scaling factor for the matrix-vector product
 * param lda - specifies the leading dimension of a
 * param inc_b - stride of vector b
 * param beta - scaling factor for vector c
 * param inc_c - stride of vector c
 * param lda - specifies the leading dimension of a
 * return norm of differences 
*/
static float check_csbmv(char uplo, blasint n, blasint k, float *alpha, blasint lda, 
    blasint inc_b, float *beta, blasint inc_c, blasint ldm)
{
    blasint i;

    // Trans param for gemv (can use any, since the input matrix is symmetric)
    char trans = 'N';

    // (lda, n) matrix for csbmv
    float a[lda * n * 2];

    // Fill triangular matrix tr_matrix for generate symmetric matrix, vectors b, c 
    rand_generate(data_csbmv.tr_matrix, n * (n + 1));
    rand_generate(data_csbmv.b, 2 * n * inc_b);
    rand_generate(data_csbmv.c, 2 * n * inc_c);

    // Copy vector c for cgemv
    for (i = 0; i < n * 2 * inc_c; i++)
        data_csbmv.C[i] = data_csbmv.c[i];

    // Generate full-storage symmetric band matrix (n, n)
    // with k super-diagonals from triangular matrix
    get_symmetric_band_matr(data_csbmv.matrix, data_csbmv.tr_matrix, n, k);

    // Transform symmetric band matrix from conventional
    // full matrix storage  to band storage for csbmv
    transform_to_band_storage(uplo, n, k, a, lda, data_csbmv.matrix, ldm);

    BLASFUNC(cgemv)(&trans, &n, &n, alpha, data_csbmv.matrix, &ldm, data_csbmv.b,
                    &inc_b, beta, data_csbmv.C, &inc_c);

    BLASFUNC(csbmv)(&uplo, &n, &k, alpha, a, &lda,
                    data_csbmv.b, &inc_b, beta, data_csbmv.c, &inc_c);

    // Find the differences between output vector caculated by csbmv and cgemv
    for (i = 0; i < n * 2 * inc_c; i++)
        data_csbmv.c[i] -= data_csbmv.C[i];

    // Find the norm of differences
    return BLASFUNC(scnrm2)(&n, data_csbmv.c, &inc_c);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is upper-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 0 
*/
CTEST(csbmv, upper_k_0_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 0;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is upper-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 1
*/
CTEST(csbmv, upper_k_1_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 1;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is upper-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, upper_k_2_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is upper-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 1
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, upper_k_2_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is upper-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 2
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, upper_k_2_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is lower-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 0
*/
CTEST(csbmv, lower_k_0_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 0;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'L';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is lower-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 1
*/
CTEST(csbmv, lower_k_1_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 1;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'L';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is lower-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, lower_k_2_inc_b_1_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'L';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is lower-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 1
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, lower_k_2_inc_b_2_inc_c_1_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 1;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'L';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test csbmv by comparing it against cgemv
 * with the following options:
 * 
 * a is lower-band-packed symmetric matrix
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 2
 * Number of super-diagonals k is 2
*/
CTEST(csbmv, lower_k_2_inc_b_2_inc_c_2_N_100)
{
    blasint N = DATASIZE, inc_b = 2, inc_c = 2;
    blasint K = 2;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'L';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/** 
 * Check if output matrix a contains any NaNs
*/
CTEST(csbmv, check_for_NaN)
{
    blasint N = DATASIZE, inc_b = 1, inc_c = 1;
    blasint K = 0;
    blasint LDA = K + 1;
    blasint LDM = N;
    char uplo = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);

    ASSERT_TRUE(norm == norm); /* NaN == NaN is false */
}

/**
 * Test error function for an invalid param uplo.
 * Uplo specifies whether a is in upper or lower band-packed storage mode.
*/
CTEST(csbmv, xerbla_uplo_invalid)
{
    blasint N = 1, inc_b = 1, inc_c = 1;
    char uplo = 'O';
    blasint K = 0;
    blasint LDA = K + 1;
    int expected_info = 1;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/** 
 * Test error function for an invalid param N -
 * number of rows and columns of A. Must be at least zero.
*/
CTEST(csbmv, xerbla_N_invalid)
{
    blasint N = INVALID, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    blasint K = 0;
    blasint LDA = K + 1;
    int expected_info = 2;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Check if N - number of rows and columns of A equal zero.
*/
CTEST(csbmv, check_N_zero)
{
    blasint N = 0, inc_b = 1, inc_c = 1;
    blasint K = 0;
    blasint LDA = K + 1;
    blasint LDM = 1;
    char uplo = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float norm;

    norm = check_csbmv(uplo, N, K, alpha, LDA, inc_b, beta, inc_c, LDM);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_TOL);
}

/**
 * Test error function for an invalid param inc_b -
 * stride of vector b. Can't be zero. 
*/
CTEST(csbmv, xerbla_inc_b_zero)
{
    blasint N = 1, inc_b = 0, inc_c = 1;
    char uplo = 'U';
    blasint K = 0;
    blasint LDA = K + 1;
    int expected_info = 8;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_c -
 * stride of vector c. Can't be zero. 
*/
CTEST(csbmv, xerbla_inc_c_zero)
{
    blasint N = 1, inc_b = 1, inc_c = 0;
    char uplo = 'U';
    blasint K = 0;
    blasint LDA = K + 1;
    int expected_info = 11;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param k -
 * number of super-diagonals of A. Must be at least zero.
*/
CTEST(csbmv, xerbla_k_invalid)
{
    blasint N = 1, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    blasint K = INVALID;
    blasint LDA = K + 1;
    int expected_info = 3;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda -
 * specifies the leading dimension of a. Must be at least (k+1).
*/
CTEST(csbmv, xerbla_lda_invalid)
{
    blasint N = 1, inc_b = 1, inc_c = 1;
    char uplo = 'U';
    blasint K = 0;
    blasint LDA = 0;
    int expected_info = 6;
    int passed;

    passed = check_badargs(uplo, N, K, LDA, inc_b, inc_c, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif
