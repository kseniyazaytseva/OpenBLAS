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

#define N 100
#define K 100
#define M 100

struct DATA_ZGEADD{
    double A[M * N * 2];
    double B[K * N * 2];
    double C_Test[M * N * 2];
    double C_Verify[M * N * 2];
};


#ifdef BUILD_COMPLEX16
static struct DATA_ZGEADD data_zgeadd;

/**
 * Generate random matrix
 */
static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * Generate identity matrix
 */
static void identity_matr_generate(double *a, blasint n, blasint ld)
{
    blasint i;
    blasint offset = 1;

    // Сomplex multipliers
    ld *= 2;
    offset *= 2;
    
    double *a_ptr = a;

    memset(a, 0.0, n * ld * sizeof(double));
    
    for (i = 0; i < n; i++) {
        a_ptr[0] = 1.0;
        a_ptr += ld + offset;
    }
}

/**
 * Find difference between two rectangle matrix
 * return norm of differences
 */
static double matrix_difference(double *a, double *b, blasint n, blasint ld)
{
    blasint i = 0;
    blasint j = 0;
    blasint inc = 1;
    double norm = 0.0;

    // Сomplex multiplier
    ld *= 2;

    double *a_ptr = a;
    double *b_ptr = b;

    for(i = 0; i < n; i++)
    {
        for (j = 0; j < ld; j++) {
            a_ptr[j] -= b_ptr[j];
        }
        norm += cblas_dnrm2(ld, a_ptr, inc);
        
        a_ptr += ld;
        b_ptr += ld;
    }
    return norm/(double)(n);
}

/**
 * Check if error function was called with expected function name
 * and param info
 * 
 * param m - number of rows of A and C
 * param n - number of columns of A and C
 * param llda - leading dimension of A
 * param lldc - leading dimension of C
 * param expected_info - expected invalid parameter number in zgeadd
 * return TRUE if everything is ok, otherwise FALSE 
 */
static int check_badargs(blasint m, blasint n, blasint llda,
                            blasint lldc, int expected_info)
{
    double a[2];
    double c[2];
    rand_generate(a, 2);
    rand_generate(c, 2);

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    set_xerbla("ZGEADD ", expected_info);

    BLASFUNC(zgeadd)(&m, &n, 
                alpha, a, &llda, 
                beta, c, &lldc);

    return check_error();
}

/**
 * C API specific function.
 * 
 * Check if error function was called with expected function name
 * and param info
 * 
 * c api param order - specifies whether A and C stored in
 * row-major order or column-major order
 * param m - number of rows of A and C
 * param n - number of columns of A and C
 * param llda - leading dimension of A
 * param lldc - leading dimension of C
 * param expected_info - expected invalid parameter number in zgeadd
 * return TRUE if everything is ok, otherwise FALSE 
 */
static int c_api_check_badargs(OPENBLAS_CONST enum CBLAS_ORDER order,
                                blasint m, blasint n, blasint llda,
                                blasint lldc, int expected_info)
{
    double a[2];
    double c[2];
    rand_generate(a, 2);
    rand_generate(c, 2);

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    set_xerbla("ZGEADD ", expected_info);

    cblas_zgeadd(order, m, n, alpha, a, llda, 
                    beta, c, lldc);

    return check_error();
}

/**
 * Test zgeadd by comparing it against zgemm.
 * The zgeadd perform operation: C:=beta*C + alpha*A
 * The zgemm perform operation: C:=alpha*op(A)*op(B) + beta*C
 * The operations are equivalent when B - identity matrix, op(x)=x
 * Compare with the following options:
 * 
 * param m - number of rows of A and C
 * param n - number of columns of A, B, C
 * param k - number of rows of B
 * param alpha - scaling factor for matrix A
 * param lda - leading dimension of A for zgemm
 * param llda - leading dimension of A for zgeadd
 * param ldb - leading dimension of B for zgemm
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C for zgemm
 * param lldc - leading dimension of C for zgeadd
 * return norm of differences
 */
static double check_zgeadd(blasint m, blasint n, blasint k,
	                        double *alpha, blasint lda, blasint llda, blasint ldb,
	                        double *beta, blasint ldc, blasint lldc)
{
    blasint i;

    // Trans param for zgemm (not transform A and C)
    char trans_a = 'N';
    char trans_b = 'N';

    // Fill matrix A, C
    rand_generate(data_zgeadd.A, lda * n * 2);
    rand_generate(data_zgeadd.C_Test, ldc * n * 2);

    // Make B identity matrix
    identity_matr_generate(data_zgeadd.B, n, ldb);

    // Copy matrix C for zgeadd
    for (i = 0; i < ldc * n * 2; i++)
        data_zgeadd.C_Verify[i] = data_zgeadd.C_Test[i];

    BLASFUNC(zgemm)(&trans_a, &trans_b, &m, &n, &k, 
                    alpha, data_zgeadd.A, &lda, data_zgeadd.B, &ldb, 
                    beta, data_zgeadd.C_Verify, &ldc);

    BLASFUNC(zgeadd)(&m, &n, alpha, data_zgeadd.A, &llda, 
                    beta, data_zgeadd.C_Test, &lldc);

    // Find the differences between output matrix caculated by zgeadd and zgemm
    return matrix_difference(data_zgeadd.C_Test, data_zgeadd.C_Verify, n, lldc);
}

/**
 * C API specific function.
 * 
 * Test zgeadd by comparing it against zgemm.
 * The zgeadd perform operation: C:=beta*C + alpha*A
 * The zgemm perform operation: C:=alpha*op(A)*op(B) + beta*C
 * The operations are equivalent when B - identity matrix, op(x)=x
 * Compare with the following options:
 * 
 * c api param order - specifies whether A and C stored in
 * row-major order or column-major order
 * 
 * c api param trans_a - specifies transpose option for matrix A.
 * c api param trans_a - specifies transpose option for matrix B.
 * row-major order or column-major order
 * param m - number of rows of A and C
 * param n - number of columns of A, B, C
 * param k - number of rows of B
 * param alpha - scaling factor for matrix A
 * param lda - leading dimension of A for zgemm
 * param llda - leading dimension of A for zgeadd
 * param ldb - leading dimension of B for zgemm
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C for zgemm
 * param lldc - leading dimension of C for zgeadd
 * return norm of differences
 */
static double c_api_check_zgeadd(OPENBLAS_CONST enum CBLAS_ORDER order,
                                    OPENBLAS_CONST enum CBLAS_TRANSPOSE trans_a,
                                    OPENBLAS_CONST enum CBLAS_TRANSPOSE trans_b,
                                    blasint m, blasint n, blasint k,
                                    double *alpha, blasint lda, blasint llda,
                                    blasint ldb, double *beta, blasint ldc,
                                    blasint lldc)
{
    blasint i;

    // Fill matrix A, C
    rand_generate(data_zgeadd.A, lda * n * 2);
    rand_generate(data_zgeadd.C_Test, ldc * n * 2);

    // Make B identity matrix
    identity_matr_generate(data_zgeadd.B, n, ldb);

    // Copy matrix C for zgeadd
    for (i = 0; i < ldc * n * 2; i++)
        data_zgeadd.C_Verify[i] = data_zgeadd.C_Test[i];

    cblas_zgemm(order, trans_a, trans_b, m, n, k, 
                    alpha, data_zgeadd.A, lda, data_zgeadd.B, ldb, 
                    beta, data_zgeadd.C_Test, ldc);

    cblas_zgeadd(order, m, n, alpha, data_zgeadd.A, llda, 
                    beta, data_zgeadd.C_Verify, lldc);
    
    // Find the differences between output matrix caculated by zgeadd and zgemm
    return matrix_difference(data_zgeadd.C_Test, data_zgeadd.C_Verify, n, lldc);
}

/**
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(zgeadd, matrix_n_100_m_100)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {3.0, 2.0};
    double beta[] = {1.0, 3.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar alpha is zero (operation is C:=beta*C)
 */
CTEST(zgeadd, matrix_n_100_m_100_alpha_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {0.0, 0.0};
    double beta[] = {2.5, 1.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar beta is zero (operation is C:=alpha*A)
 */
CTEST(zgeadd, matrix_n_100_m_100_beta_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {3.0, 1.5};
    double beta[] = {0.0, 0.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalars alpha, beta is zero (operation is C:= 0)
 */
CTEST(zgeadd, matrix_n_100_m_100_alpha_beta_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {0.0, 0.0};
    double beta[] = {0.0, 0.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(zgeadd, matrix_n_100_m_50)
{
    blasint n = N;
    blasint k = K;
    blasint m = M/2;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/** 
 * Test error function for an invalid param n -
 * number of columns of A and C
 * Must be at least zero.
 */
CTEST(zgeadd, xerbla_n_invalid)
{
    blasint n = INVALID;
    blasint m = 1;

    blasint llda = m;
    blasint lldc = m;

    int expected_info = 2;
    int passed;

    passed = check_badargs(m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/** 
 * Test error function for an invalid param m -
 * number of rows of A and C
 * Must be at least zero.
 */
CTEST(zgeadd, xerbla_m_invalid)
{
    blasint n = 1;
    blasint m = INVALID;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 1;
    int passed;

    passed = check_badargs(m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda -
 * specifies the leading dimension of A. Must be at least MAX(1, m).
 */
CTEST(zgeadd, xerbla_lda_invalid)
{
    blasint n = 1;
    blasint m = 1;

    blasint llda = INVALID;
    blasint lldc = 1;

    int expected_info = 6;
    int passed;

    passed = check_badargs(m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param ldc -
 * specifies the leading dimension of C. Must be at least MAX(1, m).
 */
CTEST(zgeadd, xerbla_ldc_invalid)
{
    blasint n = 1;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = INVALID;

    int expected_info = 8;
    int passed;

    passed = check_badargs(m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Check if n - number of columns of A, B, C equal zero.
 */
CTEST(zgeadd, n_zero)
{
    blasint n = 0;
    blasint m = 1;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Check if m - number of rows of A and C equal zero.
 */
CTEST(zgeadd, m_zero)
{
    blasint n = 1;
    blasint m = 0;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    double norm = check_zgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(zgeadd, c_api_matrix_n_100_m_100)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {2.0, 1.0};
    double beta[] = {1.0, 3.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is row-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(zgeadd, c_api_matrix_n_100_m_100_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {4.0, 1.5};
    double beta[] = {2.0, 1.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is row-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(zgeadd, c_api_matrix_n_50_m_100_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N/2;
    blasint k = K/2;
    blasint m = M;

    blasint lda = n;
    blasint ldb = k;
    blasint ldc = n;

    blasint llda = n;
    blasint lldc = n;

    double alpha[] = {3.0, 2.5};
    double beta[] = {1.0, 2.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar alpha is zero (operation is C:=beta*C)
 */
CTEST(zgeadd, c_api_matrix_n_100_m_100_alpha_zero)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {0.0, 0.0};
    double beta[] = {1.0, 1.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar beta is zero (operation is C:=alpha*A)
 */
CTEST(zgeadd, c_api_matrix_n_100_m_100_beta_zero)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {3.0, 1.5};
    double beta[] = {0.0, 0.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalars alpha, beta is zero (operation is C:= 0)
 */
CTEST(zgeadd, c_api_matrix_n_100_m_100_alpha_beta_zero)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;
    
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {0.0, 0.0};
    double beta[] = {0.0, 0.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zgeadd by comparing it against zgemm
 * with the following options:
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(zgeadd, c_api_matrix_n_100_m_50)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;

    blasint n = N;
    blasint k = K;
    blasint m = M/2;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    double alpha[] = {2.0, 3.0};
    double beta[] = {2.0, 4.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param order -
 * specifies whether A and C stored in
 * row-major order or column-major order
 */
CTEST(zgeadd, c_api_xerbla_invalid_order)
{
    CBLAS_ORDER order = INVALID;
    
    blasint n = 1;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 0;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param n -
 * number of columns of A and C.
 * Must be at least zero.
 * 
 * c api option order is column-major order
 */
CTEST(zgeadd, c_api_xerbla_n_invalid)
{
    CBLAS_ORDER order = CblasColMajor;
    
    blasint n = INVALID;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 2;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param n -
 * number of columns of A and C.
 * Must be at least zero.
 * 
 * c api option order is row-major order
 */
CTEST(zgeadd, c_api_xerbla_n_invalid_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    
    blasint n = INVALID;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 1;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param m -
 * number of rows of A and C
 * Must be at least zero.
 * 
 * c api option order is column-major order
 */
CTEST(zgeadd, c_api_xerbla_m_invalid)
{
    CBLAS_ORDER order = CblasColMajor;
    
    blasint n = 1;
    blasint m = INVALID;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 1;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param m -
 * number of rows of A and C
 * Must be at least zero.
 * 
 * c api option order is row-major order
 */
CTEST(zgeadd, c_api_xerbla_m_invalid_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    
    blasint n = 1;
    blasint m = INVALID;

    blasint llda = 1;
    blasint lldc = 1;

    int expected_info = 2;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param lda -
 * specifies the leading dimension of A. Must be at least MAX(1, m).
 * 
 * c api option order is column-major order
 */
CTEST(zgeadd, c_api_xerbla_lda_invalid)
{
    CBLAS_ORDER order = CblasColMajor;
    
    blasint n = 1;
    blasint m = 1;

    blasint llda = INVALID;
    blasint lldc = 1;

    int expected_info = 5;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param lda -
 * specifies the leading dimension of A. Must be at least MAX(1, m).
 * 
 * c api option order is row-major order
 */
CTEST(zgeadd, c_api_xerbla_lda_invalid_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    
    blasint n = 1;
    blasint m = 1;

    blasint llda = INVALID;
    blasint lldc = 1;

    int expected_info = 5;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param ldc -
 * specifies the leading dimension of C. Must be at least MAX(1, m).
 * 
 * c api option order is column-major order
 */
CTEST(zgeadd, c_api_xerbla_ldc_invalid)
{
    CBLAS_ORDER order = CblasColMajor;
    
    blasint n = 1;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = INVALID;

    int expected_info = 8;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param ldc -
 * specifies the leading dimension of C. Must be at least MAX(1, m).
 * 
 * c api option order is row-major order
 */
CTEST(zgeadd, c_api_xerbla_ldc_invalid_row_major)
{
    CBLAS_ORDER order = CblasRowMajor;
    
    blasint n = 1;
    blasint m = 1;

    blasint llda = 1;
    blasint lldc = INVALID;

    int expected_info = 8;
    int passed;

    passed = c_api_check_badargs(order, m, n, llda, lldc, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * C API specific test
 * 
 * Check if n - number of columns of A, B, C equal zero.
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 */
CTEST(zgeadd, c_api_n_zero)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;
    
    blasint n = 0;
    blasint m = 1;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Check if m - number of rows of A and C equal zero.
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 */
CTEST(zgeadd, c_api_m_zero)
{
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE trans_a = CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = CblasNoTrans;
    
    blasint n = 1;
    blasint m = 0;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    double alpha[] = {1.0, 1.0};
    double beta[] = {1.0, 1.0};

    double norm = c_api_check_zgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}
#endif