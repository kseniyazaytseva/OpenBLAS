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

struct DATA_DGEMMT
{
    double a_test[DATASIZE * DATASIZE];
    double b_test[DATASIZE * DATASIZE];
    double c_test[DATASIZE * DATASIZE];
    double c_verify[DATASIZE * DATASIZE];
    double c_gemm[DATASIZE * DATASIZE];
};

#ifdef BUILD_DOUBLE
static struct DATA_DGEMMT data_dgemmt;

/**
 * Compute gemmt via gemm since gemmt is gemm but updates only 
 * the upper or lower triangular part of the result matrix
 *
 * param api specifies tested api (C or Fortran)
 * param order specifies row or column major order (for Fortran API column major always)
 * param uplo specifies whether C’s data is stored in its upper or lower triangle
 * param transa specifies op(A), the transposition operation applied to A
 * param transb specifies op(B), the transposition operation applied to B
 * param m - number of rows of op(A), columns of op(B), and columns and rows of C
 * param k - number of columns of op(A) and rows of op(B)
 * param alpha - scaling factor for the matrix-matrix product
 * param lda - leading dimension of A
 * param ldb - leading dimension of B
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C
 */
static void dgemmt_trusted(char api, enum CBLAS_ORDER order, char uplo, char transa, 
                           char transb, blasint m, blasint k, double alpha, blasint lda, 
                           blasint ldb, double beta, blasint ldc)
{
    blasint i, j;

    if(api == 'F')
        BLASFUNC(dgemm)(&transa, &transb, &m, &m, &k, &alpha, data_dgemmt.a_test, &lda,
                        data_dgemmt.b_test, &ldb, &beta, data_dgemmt.c_gemm, &ldc);
    else
        cblas_dgemm(order, transa, transb, m, m, k, alpha, data_dgemmt.a_test, lda,
                data_dgemmt.b_test, ldb, beta, data_dgemmt.c_gemm, ldc);

    if (uplo == 'L' || uplo == CblasLower)
    {
        for (i = 0; i < m; i++)
            for (j = i; j < m; j++)
                data_dgemmt.c_verify[i * ldc + j] =
                    data_dgemmt.c_gemm[i * ldc + j];
    } else {
        for (i = 0; i < m; i++)
            for (j = 0; j <= i; j++)
                data_dgemmt.c_verify[i * ldc + j] =
                    data_dgemmt.c_gemm[i * ldc + j];
    }
}

static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * Comapare results computed by dgemmt and dgemmt_trusted
 *
 * param api specifies tested api (C or Fortran)
 * param order specifies row or column major order (for Fortran API column major always)
 * param uplo specifies whether C’s data is stored in its upper or lower triangle
 * param transa specifies op(A), the transposition operation applied to A
 * param transb specifies op(B), the transposition operation applied to B
 * param m - number of rows of op(A), columns of op(B), and columns and rows of C
 * param k - number of columns of op(A) and rows of op(B)
 * param alpha - scaling factor for the matrix-matrix product
 * param lda - leading dimension of A
 * param ldb - leading dimension of B
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C
 * return norm of differences
 */
static double check_dgemmt(char api, enum CBLAS_ORDER order, char uplo, char transa, 
                          char transb, blasint m, blasint k, double alpha, blasint lda, 
                          blasint ldb, double beta, blasint ldc)
{
    blasint i;
    blasint b_cols;
    blasint a_cols;
    blasint inc = 1;
    blasint size_c = m * ldc;

    if(order == CblasColMajor){
        if (transa == 'T' || transa == 'C' || 
            transa == CblasTrans || transa == CblasConjTrans) 
            a_cols = m;
        else a_cols = k;

        if (transb == 'T' || transb == 'C' || 
            transb == CblasTrans || transb == CblasConjTrans) 
            b_cols = k;
        else b_cols = m;
    } else {
        if (transa == 'T' || transa == 'C' || 
            transa == CblasTrans || transa == CblasConjTrans) 
            a_cols = k;
        else a_cols = m;

        if (transb == 'T' || transb == 'C' ||
            transb == CblasTrans || transb == CblasConjTrans) 
            b_cols = m;
        else b_cols = k;
    }

    rand_generate(data_dgemmt.a_test, a_cols * lda);
    rand_generate(data_dgemmt.b_test, b_cols * ldb);
    rand_generate(data_dgemmt.c_test, m * ldc);

    for (i = 0; i < m * ldc; i++)
        data_dgemmt.c_gemm[i] = data_dgemmt.c_verify[i] = data_dgemmt.c_test[i];

    dgemmt_trusted(api, order, uplo, transa, transb, m, k, alpha, lda, ldb, beta, ldc);

    if (api == 'F')
        BLASFUNC(dgemmt)(&uplo, &transa, &transb, &m, &k, &alpha, data_dgemmt.a_test,
                         &lda, data_dgemmt.b_test, &ldb, &beta, data_dgemmt.c_test, &ldc);
    else
        cblas_dgemmt(order, uplo, transa, transb, m, k, alpha, data_dgemmt.a_test, lda,
                    data_dgemmt.b_test, ldb, beta, data_dgemmt.c_test, ldc);

    for (i = 0; i < m * ldc; i++)
        data_dgemmt.c_verify[i] -= data_dgemmt.c_test[i];

    return BLASFUNC(dnrm2)(&size_c, data_dgemmt.c_verify, &inc) / size_c;
}

/**
 * Check if error function was called with expected function name
 * and param info
 *
 * param uplo specifies whether C’s data is stored in its upper or lower triangle
 * param transa specifies op(A), the transposition operation applied to A
 * param transb specifies op(B), the transposition operation applied to B
 * param m - number of rows of op(A), columns of op(B), and columns and rows of C
 * param k - number of columns of op(A) and rows of op(B)
 * param lda - leading dimension of A
 * param ldb - leading dimension of B
 * param ldc - leading dimension of C
 * param expected_info - expected invalid parameter number in dgemmt
 * return TRUE if everything is ok, otherwise FALSE
 */
static int check_badargs(char api, enum CBLAS_ORDER order, char uplo, char transa, 
                         char transb, blasint m, blasint k, blasint lda, blasint ldb,
                         blasint ldc, int expected_info)
{
    double alpha = 1.0;
    double beta = 0.0;

    set_xerbla("DGEMMT ", expected_info);

    if (api == 'F')
        BLASFUNC(dgemmt)(&uplo, &transa, &transb, &m, &k, &alpha, data_dgemmt.a_test,
                         &lda, data_dgemmt.b_test, &ldb, &beta, data_dgemmt.c_test, &ldc);
    else
        cblas_dgemmt(order, uplo, transa, transb, m, k, alpha, data_dgemmt.a_test, lda,
                    data_dgemmt.b_test, ldb, beta, data_dgemmt.c_test, ldc);

    return check_error();
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, upper_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, upper_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 100;
    char transa = 'T', transb = 'N';
    char uplo = 'U';
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, upper_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'T';
    char uplo = 'U';
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, upper_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'T', transb = 'T';
    char uplo = 'U';
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, upper_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its upper triangle part
 * beta = 1.0
 */
CTEST(dgemmt, upper_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, lower_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'L';
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, lower_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 100;
    char transa = 'T', transb = 'N';
    char uplo = 'L';
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, lower_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'T';
    char uplo = 'L';
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, lower_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'T', transb = 'T';
    char uplo = 'L';
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, lower_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'L';
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * C’s data is stored in its lower triangle part
 * beta = 1.0
 */
CTEST(dgemmt, lower_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'L';
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('F', CblasColMajor, uplo, transa, transb,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_colmajor_upper_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_colmajor_upper_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 100;
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, c_api_colmajor_upper_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, c_api_colmajor_upper_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, c_api_colmajor_upper_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its upper triangle part
 * beta = 1.0
 */
CTEST(dgemmt, c_api_colmajor_upper_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_colmajor_lower_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_colmajor_lower_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 100;
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, c_api_colmajor_lower_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasNoTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, c_api_colmajor_lower_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, c_api_colmajor_lower_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Column Major
 * C’s data is stored in its lower triangle part
 * beta = 1.0
 */
CTEST(dgemmt, c_api_colmajor_lower_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('C', CblasColMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_rowmajor_upper_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_rowmajor_upper_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 100, ldb = 100, ldc = 100;
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, c_api_rowmajor_upper_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 100, ldb = 100, ldc = 50;
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, c_api_rowmajor_upper_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, c_api_rowmajor_upper_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its upper triangle part
 * beta = 1.0
 */
CTEST(dgemmt, c_api_rowmajor_upper_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_rowmajor_lower_M_50_K_50_a_notrans_b_notrans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * A transposed
 * B not transposed
 */
CTEST(dgemmt, c_api_rowmajor_lower_M_100_K_50_a_trans_b_notrans)
{
    blasint M = 100, K = 50;
    blasint lda = 100, ldb = 100, ldc = 100;
    double alpha = 1.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * A not transposed
 * B transposed
 */
CTEST(dgemmt, c_api_rowmajor_lower_M_50_K_100_a_notrans_b_trans)
{
    blasint M = 50, K = 100;
    blasint lda = 100, ldb = 100, ldc = 50;
    double alpha = 1.0;
    double beta = 0.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasNoTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * A transposed
 * B transposed
 */
CTEST(dgemmt, c_api_rowmajor_lower_M_50_K_50_a_trans_b_trans)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 1.5;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasTrans, CblasTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * alpha = 0.0
 */
CTEST(dgemmt, c_api_rowmajor_lower_alpha_zero)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 0.0;
    double beta = 2.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dgemmt by comparing it against dgemm
 * with the following options:
 *
 * Row Major
 * C’s data is stored in its lower triangle part
 * beta = 1.0
 */
CTEST(dgemmt, c_api_rowmajor_lower_beta_one)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    double alpha = 2.0;
    double beta = 1.0;

    double norm = check_dgemmt('C', CblasRowMajor, CblasLower, CblasNoTrans, CblasNoTrans,
                              M, K, alpha, lda, ldb, beta, ldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param uplo.
 * Must be upper (U) or lower (L).
 */
CTEST(dgemmt, xerbla_uplo_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'O';
    int expected_info = 1;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param transa.
 * Must be trans (T/C) or no-trans (N/R).
 */
CTEST(dgemmt, xerbla_transa_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'O', transb = 'N';
    char uplo = 'U';
    int expected_info = 2;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param transb.
 * Must be trans (T/C) or no-trans (N/R).
 */
CTEST(dgemmt, xerbla_transb_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'O';
    char uplo = 'U';
    int expected_info = 3;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param M.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_m_invalid)
{
    blasint M = -1, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    int expected_info = 4;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param K.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_k_invalid)
{
    blasint M = 50, K = -1;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    int expected_info = 5;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param lda.
 * Must be must be at least K if matrix A transposed.
 */
CTEST(dgemmt, xerbla_lda_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 100, ldc = 50;
    char transa = 'T', transb = 'N';
    char uplo = 'U';
    int expected_info = 8;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param ldb.
 * Must be must be at least K if matrix B not transposed.
 */
CTEST(dgemmt, xerbla_ldb_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'N', transb = 'N';
    char uplo = 'U';
    int expected_info = 10;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Fortran API specific test
 * Test error function for an invalid param ldc.
 * Must be must be at least M.
 */
CTEST(dgemmt, xerbla_ldc_invalid)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    char transa = 'T', transb = 'N';
    char uplo = 'U';
    int expected_info = 13;
    int passed;

    passed = check_badargs('F', CblasColMajor, uplo, transa, transb,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test.
 * Test error function for an invalid param order.
 * Must be column or row major.
 */
CTEST(dgemmt, xerbla_c_api_major_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 0;
    int passed;

    passed = check_badargs('C', 'O', CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param uplo.
 * Must be upper or lower.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_uplo_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 1;
    int passed;

    passed = check_badargs('C', CblasColMajor, 'O', CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param transa.
 * Must be trans or no-trans.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_transa_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 2;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, 'O', CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param transb.
 * Must be trans or no-trans.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_transb_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 3;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasNoTrans, 'O',
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param M.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_m_invalid)
{
    blasint M = -1, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 4;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param K.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_k_invalid)
{
    blasint M = 50, K = -1;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 5;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param lda.
 * Must be must be at least K if matrix A transposed.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_lda_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 100, ldc = 50;
    int expected_info = 8;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param ldb.
 * Must be must be at least K if matrix B not transposed.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_ldb_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 10;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Column Major
 * Test error function for an invalid param ldc.
 * Must be must be at least M.
 */
CTEST(dgemmt, xerbla_c_api_colmajor_ldc_invalid)
{
    blasint M = 100, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 13;
    int passed;

    passed = check_badargs('C', CblasColMajor, CblasUpper, CblasTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param uplo.
 * Must be upper or lower.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_uplo_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 1;
    int passed;

    passed = check_badargs('C', CblasRowMajor, 'O', CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param transa.
 * Must be trans or no-trans.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_transa_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 2;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, 'O', CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param transb.
 * Must be trans or no-trans.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_transb_invalid)
{
    blasint M = 50, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 3;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasNoTrans, 'O',
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param M.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_m_invalid)
{
    blasint M = -1, K = 50;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 4;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param K.
 * Must be positive.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_k_invalid)
{
    blasint M = 50, K = -1;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 5;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param lda.
 * Must be must be at least K if matrix A transposed.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_lda_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 8;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param ldb.
 * Must be must be at least K if matrix B transposed.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_ldb_invalid)
{
    blasint M = 50, K = 100;
    blasint lda = 50, ldb = 50, ldc = 50;
    int expected_info = 10;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasTrans, CblasTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test. Row Major
 * Test error function for an invalid param ldc.
 * Must be must be at least M.
 */
CTEST(dgemmt, xerbla_c_api_rowmajor_ldc_invalid)
{
    blasint M = 100, K = 50;
    blasint lda = 100, ldb = 100, ldc = 50;
    int expected_info = 13;
    int passed;

    passed = check_badargs('C', CblasRowMajor, CblasUpper, CblasTrans, CblasNoTrans,
                            M, K, lda, ldb, ldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif