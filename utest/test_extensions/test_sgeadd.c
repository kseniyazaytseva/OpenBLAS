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

struct DATA_SGEADD{
    float A[M * N];
    float B[K * N];
    float C_Test[M * N];
    float C_Verify[M * N];
};


#ifdef BUILD_SINGLE
static struct DATA_SGEADD data_sgeadd;

/**
 * Generate random matrix
 */
static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
 * Generate identity matrix
 */
static void identity_matr_generate(float *a, blasint n, blasint ld)
{
    blasint i;
    blasint offset = 1;

    float *a_ptr = a;

    memset(a, 0.0f, n * ld * sizeof(float));
    
    for (i = 0; i < n; i++) {
        a_ptr[0] = 1.0f;
        a_ptr += ld + offset;
    }
}

/**
 * Find difference between two rectangle matrix
 * return norm of differences
 */
static float matrix_difference(float *a, float *b, blasint n, blasint ld)
{
    blasint i = 0;
    blasint j = 0;
    blasint inc = 1;
    float norm = 0.0f;

    float *a_ptr = a;
    float *b_ptr = b;

    for(i = 0; i < n; i++)
    {
        for (j = 0; j < ld; j++) {
            a_ptr[j] -= b_ptr[j];
        }
        norm += cblas_snrm2(ld, a_ptr, inc);
        
        a_ptr += ld;
        b_ptr += ld;
    }
    return norm/(float)(n);
}

/**
 * Check if error function was called with expected function name
 * and param info
 * 
 * param m - number of rows of A and C
 * param n - number of columns of A and C
 * param llda - leading dimension of A
 * param lldc - leading dimension of C
 * param expected_info - expected invalid parameter number in sgeadd
 * return TRUE if everything is ok, otherwise FALSE 
 */
static int check_badargs(blasint m, blasint n, blasint llda,
                            blasint lldc, int expected_info)
{
    float a;
    float c;
    rand_generate(&a, 1);
    rand_generate(&c, 1);

    float alpha = 1.0f;
    float beta = 1.0f;

    set_xerbla("SGEADD ", expected_info);

    BLASFUNC(sgeadd)(&m, &n, 
                &alpha, &a, &llda, 
                &beta, &c, &lldc);

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
 * param expected_info - expected invalid parameter number in sgeadd
 * return TRUE if everything is ok, otherwise FALSE 
 */
static int c_api_check_badargs(OPENBLAS_CONST enum CBLAS_ORDER order,
                                blasint m, blasint n, blasint llda,
                                blasint lldc, int expected_info)
{
    float a;
    float c;
    rand_generate(&a, 1);
    rand_generate(&c, 1);

    float alpha = 1.0f;
    float beta = 1.0f;

    set_xerbla("SGEADD ", expected_info);

    cblas_sgeadd(order, m, n, alpha, &a, llda, 
                    beta, &c, lldc);

    return check_error();
}

/**
 * Test sgeadd by comparing it against sgemm.
 * The sgeadd perform operation: C:=beta*C + alpha*A
 * The sgemm perform operation: C:=alpha*op(A)*op(B) + beta*C
 * The operations are equivalent when B - identity matrix, op(x)=x
 * Compare with the following options:
 * 
 * param m - number of rows of A and C
 * param n - number of columns of A, B, C
 * param k - number of rows of B
 * param alpha - scaling factor for matrix A
 * param lda - leading dimension of A for sgemm
 * param llda - leading dimension of A for sgeadd
 * param ldb - leading dimension of B for sgemm
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C for sgemm
 * param lldc - leading dimension of C for sgeadd
 * return norm of differences
 */
static float check_sgeadd(blasint m, blasint n, blasint k,
	                        float alpha, blasint lda, blasint llda, blasint ldb,
	                        float beta, blasint ldc, blasint lldc)
{
    blasint i;

    // Trans param for sgemm (not transform A and C)
    char trans_a = 'N';
    char trans_b = 'N';

    // Fill matrix A, C
    rand_generate(data_sgeadd.A, lda * n);
    rand_generate(data_sgeadd.C_Test, ldc * n);

    // Make B identity matrix
    identity_matr_generate(data_sgeadd.B, n, ldb);

    // Copy matrix C for sgeadd
    for (i = 0; i < ldc * n; i++)
        data_sgeadd.C_Verify[i] = data_sgeadd.C_Test[i];

    BLASFUNC(sgemm)(&trans_a, &trans_b, &m, &n, &k, 
                    &alpha, data_sgeadd.A, &lda, data_sgeadd.B, &ldb, 
                    &beta, data_sgeadd.C_Verify, &ldc);

    BLASFUNC(sgeadd)(&m, &n, &alpha, data_sgeadd.A, &llda, 
                    &beta, data_sgeadd.C_Test, &lldc);

    // Find the differences between output matrix caculated by sgeadd and sgemm
    return matrix_difference(data_sgeadd.C_Test, data_sgeadd.C_Verify, n, lldc);
}

/**
 * C API specific function.s
 * 
 * Test sgeadd by comparing it against sgemm.
 * The sgeadd perform operation: C:=beta*C + alpha*A
 * The sgemm perform operation: C:=alpha*op(A)*op(B) + beta*C
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
 * param lda - leading dimension of A for sgemm
 * param llda - leading dimension of A for sgeadd
 * param ldb - leading dimension of B for sgemm
 * param beta - scaling factor for matrix C
 * param ldc - leading dimension of C for sgemm
 * param lldc - leading dimension of C for sgeadd
 * return norm of differences
 */
static float c_api_check_sgeadd(OPENBLAS_CONST enum CBLAS_ORDER order,
                                    OPENBLAS_CONST enum CBLAS_TRANSPOSE trans_a,
                                    OPENBLAS_CONST enum CBLAS_TRANSPOSE trans_b,
                                    blasint m, blasint n, blasint k,
                                    float alpha, blasint lda, blasint llda,
                                    blasint ldb, float beta, blasint ldc,
                                    blasint lldc)
{
    blasint i;

    // Fill matrix A, C
    rand_generate(data_sgeadd.A, lda * n);
    rand_generate(data_sgeadd.C_Test, ldc * n);

    // Make B identity matrix
    identity_matr_generate(data_sgeadd.B, n, ldb);

    // Copy matrix C for sgeadd
    for (i = 0; i < ldc * n; i++)
        data_sgeadd.C_Verify[i] = data_sgeadd.C_Test[i];

    cblas_sgemm(order, trans_a, trans_b, m, n, k, 
                    alpha, data_sgeadd.A, lda, data_sgeadd.B, ldb, 
                    beta, data_sgeadd.C_Test, ldc);

    cblas_sgeadd(order, m, n, alpha, data_sgeadd.A, llda, 
                    beta, data_sgeadd.C_Verify, lldc);
    
    // Find the differences between output matrix caculated by sgeadd and sgemm
    return matrix_difference(data_sgeadd.C_Test, data_sgeadd.C_Verify, n, lldc);
}

/**
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(sgeadd, matrix_n_100_m_100)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    float alpha = 3.0f;
    float beta = 3.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar alpha is zero (operation is C:=beta*C)
 */
CTEST(sgeadd, matrix_n_100_m_100_alpha_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    float alpha = 0.0f;
    float beta = 2.5f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalar beta is zero (operation is C:=alpha*A)
 */
CTEST(sgeadd, matrix_n_100_m_100_beta_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    float alpha = 3.0f;
    float beta = 0.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 * Scalars alpha, beta is zero (operation is C:= 0)
 */
CTEST(sgeadd, matrix_n_100_m_100_alpha_beta_zero)
{
    blasint n = N;
    blasint k = K;
    blasint m = M;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    float alpha = 0.0f;
    float beta = 0.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(sgeadd, matrix_n_100_m_50)
{
    blasint n = N;
    blasint k = K;
    blasint m = M/2;

    blasint lda = m;
    blasint ldb = k;
    blasint ldc = m;

    blasint llda = m;
    blasint lldc = m;

    float alpha = 1.0f;
    float beta = 1.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/** 
 * Test error function for an invalid param n -
 * number of columns of A and C
 * Must be at least zero.
 */
CTEST(sgeadd, xerbla_n_invalid)
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
CTEST(sgeadd, xerbla_m_invalid)
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
CTEST(sgeadd, xerbla_lda_invalid)
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
CTEST(sgeadd, xerbla_ldc_invalid)
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
CTEST(sgeadd, n_zero)
{
    blasint n = 0;
    blasint m = 1;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    float alpha = 1.0f;
    float beta = 1.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Check if m - number of rows of A and C equal zero.
 */
CTEST(sgeadd, m_zero)
{
    blasint n = 1;
    blasint m = 0;
    blasint k = 1;

    blasint llda = 1;
    blasint lldc = 1;

    blasint lda = 1;
    blasint ldb = 1;
    blasint ldc = 1;

    float alpha = 1.0f;
    float beta = 1.0f;

    float norm = check_sgeadd(m, n, k, alpha,
                        lda, llda, ldb,
                        beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * c api option order is column-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(sgeadd, c_api_matrix_n_100_m_100)
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

    float alpha = 2.0f;
    float beta = 3.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * c api option order is row-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 100, number of colums is 100
 * For C number of rows is 100, number of colums is 100
 */
CTEST(sgeadd, c_api_matrix_n_100_m_100_row_major)
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

    float alpha = 4.0f;
    float beta = 2.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * c api option order is row-major order
 * c api option trans_a is no transpose
 * c api option trans_b is no transpose
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(sgeadd, c_api_matrix_n_50_m_100_row_major)
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

    float alpha = 3.0f;
    float beta = 1.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
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
CTEST(sgeadd, c_api_matrix_n_100_m_100_alpha_zero)
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

    float alpha = 0.0f;
    float beta = 1.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
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
CTEST(sgeadd, c_api_matrix_n_100_m_100_beta_zero)
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

    float alpha = 3.0f;
    float beta = 0.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
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
CTEST(sgeadd, c_api_matrix_n_100_m_100_alpha_beta_zero)
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

    float alpha = 0.0f;
    float beta = 0.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test sgeadd by comparing it against sgemm
 * with the following options:
 * 
 * For A number of rows is 50, number of colums is 100
 * For C number of rows is 50, number of colums is 100
 */
CTEST(sgeadd, c_api_matrix_n_100_m_50)
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

    float alpha = 3.0f;
    float beta = 4.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test error function for an invalid param order -
 * specifies whether A and C stored in
 * row-major order or column-major order
 */
CTEST(sgeadd, c_api_xerbla_invalid_order)
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
CTEST(sgeadd, c_api_xerbla_n_invalid)
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
CTEST(sgeadd, c_api_xerbla_n_invalid_row_major)
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
CTEST(sgeadd, c_api_xerbla_m_invalid)
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
CTEST(sgeadd, c_api_xerbla_m_invalid_row_major)
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
CTEST(sgeadd, c_api_xerbla_lda_invalid)
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
CTEST(sgeadd, c_api_xerbla_lda_invalid_row_major)
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
CTEST(sgeadd, c_api_xerbla_ldc_invalid)
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
CTEST(sgeadd, c_api_xerbla_ldc_invalid_row_major)
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
CTEST(sgeadd, c_api_n_zero)
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

    float alpha = 1.0f;
    float beta = 1.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
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
CTEST(sgeadd, c_api_m_zero)
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

    float alpha = 1.0f;
    float beta = 1.0f;

    float norm = c_api_check_sgeadd(order, trans_a, trans_b, 
                                m, n, k, alpha,
                                lda, llda, ldb,
	                            beta, ldc, lldc);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif