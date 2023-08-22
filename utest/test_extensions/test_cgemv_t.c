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
#define M 100
#define INCREMENT 2

struct DATA_CGEMV_T{
    float A_test[N * M * 2];
    float A_verify[N * M * 2];
    float y_test[M * INCREMENT * 2];
    float y_verify[M * INCREMENT * 2];
    float x_test[M * INCREMENT * 2];
    float x_verify[M * INCREMENT * 2];
};

// SINGLE_EPS_ZGEMV = MAX_VAL * NUMBER OF OPERATIONS * FLT_EPSILON
// SINGLE_EPS_ZGEMV = 5.0 * O(100 * 100) * 1.19e-07 = 5*e-03
#define SINGLE_EPS_ZGEMV 5e-03

#ifdef BUILD_COMPLEX
static struct DATA_CGEMV_T data_cgemv_t;

static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
 * Complex conjugate operation for vector
 * 
 * param n specifies number of elements in vector x
 * param inc_x specifies increment of vector x
 * param x specifies buffer holding vector x
 */
static void conj_vec(blasint n, blasint inc_x, float *x)
{
    blasint i;
    blasint offset = 2;

    float *x_ptr = x;

    for (i = 0; i < n; i++)
    {
        x_ptr[1] *= (-1);
        x_ptr +=  offset * inc_x;
    }
}

/**
 * Find product of matrix-vector multiplication
 * 
 * param n specifies number of columns of A
 * param m specifies number of rows of A and size of vector x
 * param lda specifies leading dimension of A
 * param inc_x specifies increment of vector x
 */
static void matrix_vector_product(blasint n, blasint m, blasint lda, blasint inc_x)
{
    blasint i;
    float *a_ptr = data_cgemv_t.A_verify;
    float *x_ptr = data_cgemv_t.x_test;
    float *x_res = data_cgemv_t.x_verify;

    openblas_complex_float result;

    for (i = 0; i < n * inc_x; i+= inc_x)
    {
        result = cblas_cdotu(lda, a_ptr, 1, x_ptr, inc_x);
        x_res[0] = CREAL(result);
        x_res[1] = CIMAG(result);
        a_ptr += lda * 2;
        x_res += 2 * inc_x;
    }
}

/**
 * Test cgemv by comparing it against comatcopy, caxpby and
 * reference func matrix_vector_product
 * 
 * comatcopy perform operation: op(A)
 * matrix_vector_product perform operation: A*x
 * caxpby perform operation: alpha*x + beta*y
 *
 * param api specifies tested api (C or Fortran)
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation 
 * applied to the matrix A
 * param m specifies number of rows of A
 * param n specifies number of columns of A
 * param alpha specifies scalar alpha
 * param lda specifies leading dimension of the matrix A
 * param inc_x specifies increment for vector x
 * param beta specifies scalar beta
 * param inc_y specifies increment for vector y
 * return norm of difference between cgemv and result of reference funcs
 */
static float check_cgemv(char api, char order, char trans, blasint m, blasint n, float *alpha, 
                            blasint lda, blasint inc_x, float *beta, blasint inc_y)
{
    blasint i;

    enum CBLAS_ORDER corder;
    enum CBLAS_TRANSPOSE ctrans;

    // Transpose parameters for comatcopy
    // cgemv_t perform operation on transposed matrix, no need to transpose A_verify
    char trans_copy;
    char ctrans_copy;

    // Param alpha for comatcopy, scale on alpha perform caxpby
    float alpha_one[] = {1.0f, 0.0f};

    memset(data_cgemv_t.x_verify, 0.0f, m * inc_x * 2 * sizeof(float));

    // Fill matrix A, vectors x, y    
    rand_generate(data_cgemv_t.A_test, lda * n * 2);
    rand_generate(data_cgemv_t.x_test, m * inc_x * 2);
    rand_generate(data_cgemv_t.y_test, m * inc_y * 2);

    // Copy vector y for reference funcs
    for (int i = 0; i < m * inc_y * 2; i++) {
        data_cgemv_t.y_verify[i] = data_cgemv_t.y_test[i];
    }

    if (api == 'F') {
        if (trans == 'T') trans_copy = 'N';
        if (trans == 'C') trans_copy = 'R';
        if (trans == 'U') trans_copy = 'R';
        if (trans == 'D') trans_copy = 'N';

        // Perform operation: op(A)
        BLASFUNC(comatcopy)(&order, &trans_copy, &m, &n, alpha_one, data_cgemv_t.A_test, &lda, data_cgemv_t.A_verify, &lda);

        // Find A*x
        matrix_vector_product(n, m, lda, inc_x);

        // Find conj(x)
        if (trans == 'U' || trans == 'D') {
            conj_vec(m, inc_x, data_cgemv_t.x_verify);
        }

        // Find alpha*x+beta*y
        BLASFUNC(caxpby)(&n, alpha, data_cgemv_t.x_verify, &inc_x, beta, data_cgemv_t.y_verify, &inc_y);

        BLASFUNC(cgemv)(&trans, &m, &n, alpha, data_cgemv_t.A_test, 
                        &lda, data_cgemv_t.x_test, &inc_x, beta, data_cgemv_t.y_test, &inc_y);
    }
    else {
        if (order == 'C') corder = CblasColMajor;
        if (order == 'R') corder = CblasRowMajor;
        if (trans == 'T') {ctrans = CblasTrans; ctrans_copy = (corder == CblasRowMajor) ? CblasTrans : CblasNoTrans;}
        if (trans == 'N') {ctrans = CblasNoTrans; ctrans_copy = (corder == CblasRowMajor) ? CblasNoTrans : CblasTrans;}
        if (trans == 'C') {ctrans = CblasConjTrans; ctrans_copy = (corder == CblasRowMajor) ? CblasConjTrans : CblasConjNoTrans;}
        if (trans == 'R') {ctrans = CblasConjNoTrans; ctrans_copy = (corder == CblasRowMajor) ? CblasConjNoTrans : CblasConjTrans;}

        // Perform operation: op(A)
        cblas_comatcopy(corder, ctrans_copy, m, n, alpha_one, data_cgemv_t.A_test, lda, data_cgemv_t.A_verify, lda);

        // Find A*x
        matrix_vector_product(n, m, lda, inc_x);

        // Find alpha*x+beta*y
        cblas_caxpby(n, alpha, data_cgemv_t.x_verify, inc_x, beta, data_cgemv_t.y_verify, inc_y);

        cblas_cgemv(corder, ctrans, m, n, alpha, data_cgemv_t.A_test, 
                        lda, data_cgemv_t.x_test, inc_x, beta, data_cgemv_t.y_test, inc_y);
    }

    // Find the differences between output vector caculated by cgemv and reference funcs
    for (i = 0; i < m * inc_y * 2; i++)
        data_cgemv_t.y_test[i] -= data_cgemv_t.y_verify[i];

    // Find the norm of differences
    return cblas_scnrm2(m, data_cgemv_t.y_test, inc_y);
}

/**
 * Check if error function was called with expected function name
 * and param info
 *
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation 
 * applied to the matrix A
 * param m specifies number of rows of A
 * param n specifies number of columns of A
 * param lda specifies leading dimension of the matrix A
 * param inc_x specifies increment for vector x
 * param inc_y specifies increment for vector y
 * param expected_info - expected invalid parameter number
 * return TRUE if everything is ok, otherwise FALSE
 */
static int check_badargs(char order, char trans, blasint m, blasint n, 
                        blasint lda, blasint inc_x, blasint inc_y, int expected_info)
{
    float alpha[] = {1.0f, 1.0f};
    float a[] = {1.0f, 1.0f};
    float x[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float y[] = {1.0f, 1.0f};

    set_xerbla("CGEMV ", expected_info);

    BLASFUNC(cgemv)(&trans, &m, &n, alpha, a, &lda, x, &inc_x, beta, y, &inc_y);

    return check_error();
}

/**
 * C API specific function
 * 
 * Check if error function was called with expected function name
 * and param info
 *
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation 
 * applied to the matrix A
 * param m specifies number of rows of A
 * param n specifies number of columns of A
 * param lda specifies leading dimension of the matrix A
 * param inc_x specifies increment for vector x
 * param inc_y specifies increment for vector y
 * param expected_info - expected invalid parameter number
 * return TRUE if everything is ok, otherwise FALSE
 */
static int c_api_check_badargs(CBLAS_ORDER corder, CBLAS_TRANSPOSE ctrans, blasint m, blasint n, 
                        blasint lda, blasint inc_x, blasint inc_y, int expected_info)
{
    float alpha[] = {1.0f, 1.0f};
    float a[] = {1.0f, 1.0f};
    float x[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};
    float y[] = {1.0f, 1.0f};

    set_xerbla("CGEMV ", expected_info);

    cblas_cgemv(corder, ctrans, m, n, alpha, a, lda, x, inc_x, beta, y, inc_y);

    return check_error();
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * inc x = 2, inc y = 1
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 2.0f, beta_i = 2.0f
 */
CTEST(cgemv, colmajor_trans_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'T';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 2.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * inc x = 2, inc y = 1
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 2.0f, beta_i = 2.0f
 */
CTEST(cgemv, colmajor_trans_col_100_row_100_inc_x_2_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'T';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {2.0f, 2.0f};

    blasint inc_x = 2;
    blasint inc_y = 1;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate A
 * Square matrix
 * inc x = 1, inc y = 1
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 2.0f, beta_i = 1.0f
 */
CTEST(cgemv, colmajor_conjtrans_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'C';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {2.0f, 1.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate A
 * Square matrix
 * inc x = 1, inc y = 2
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 2.0f, beta_i = 1.0f
 */
CTEST(cgemv, colmajor_conjtrans_col_100_row_100_inc_x_1_y_2)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'C';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    blasint inc_x = 1;
    blasint inc_y = 2;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and x conjugate
 * Square matrix
 * inc x = 1, inc y = 1
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 2.0f, beta_i = 1.0f
 */
CTEST(cgemv, colmajor_trans_x_conj_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'U';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and x conjugate
 * Square matrix
 * inc x = 2, inc y = 2
 * alpha_r = 1.0f, alpha_i = 2.0f
 * beta_r = 1.0f, beta_i = 1.0f
 */
CTEST(cgemv, colmajor_trans_x_conj_col_100_row_100_inc_x_2_y_2)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'U';

    float alpha[] = {1.0f, 2.0f};
    float beta[] = {1.0f, 1.0f};

    blasint inc_x = 2;
    blasint inc_y = 2;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}


/**
 * FORTRAN API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition, conjugate A, conjugate x
 * Square matrix
 * inc x = 2, inc y = 2
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 2.0f
 */
CTEST(cgemv, colmajor_conjtrans_x_conj_col_100_row_100_inc_x_1_y_2)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'D';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 2.0f};

    blasint inc_x = 1;
    blasint inc_y = 2;

    float norm = check_cgemv('F', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition, conjugate A, conjugate x
 * Square matrix
 * inc x = 2, inc y = 1
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 2.0f
 */
CTEST(cgemv, c_api_colmajor_trans_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'T';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 2.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate A
 * Square matrix
 * inc x = 2, inc y = 1
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 2.0f
 */
CTEST(cgemv, c_api_colmajor_conjtrans_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'C';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 2.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate A
 * Square matrix
 * inc x = 1, inc y = 2
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 2.0f
 */
CTEST(cgemv, c_api_colmajor_conjtrans_col_100_row_100_inc_x_1_y_2)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'C';
    char trans = 'C';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 2.0f};

    blasint inc_x = 1;
    blasint inc_y = 2;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Square matrix
 * inc x = 1, inc y = 1
 * alpha_r = 2.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 1.0f
 */
CTEST(cgemv, c_api_rowmajor_notrans_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'R';
    char trans = 'N';

    float alpha[] = {2.0f, 1.0f};
    float beta[] = {1.0f, 1.0f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Row Major
 * No trans
 * Square matrix
 * inc x = 2, inc y = 2
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 3.0f, beta_i = 2.0f
 */
CTEST(cgemv, c_api_rowmajor_notrans_col_100_row_100_inc_x_2_y_2)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'R';
    char trans = 'N';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {3.0f, 1.0f};

    blasint inc_x = 2;
    blasint inc_y = 2;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Conjugate
 * Square matrix
 * inc x = 1, inc y = 1
 * alpha_r = 1.0f, alpha_i = 3.0f
 * beta_r = 1.0f, beta_i = 2.5f
 */
CTEST(cgemv, c_api_rowmajor_conj_col_100_row_100_inc_x_1_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'R';
    char trans = 'R';

    float alpha[] = {1.0f, 3.0f};
    float beta[] = {1.0f, 2.5f};

    blasint inc_x = 1;
    blasint inc_y = 1;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * C API specific test
 * 
 * Test cgemv by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Conjugate
 * Square matrix
 * inc x = 2, inc y = 1
 * alpha_r = 1.0f, alpha_i = 1.0f
 * beta_r = 1.0f, beta_i = 1.5f
 */
CTEST(cgemv, c_api_rowmajor_conj_col_100_row_100_inc_x_2_y_1)
{
    blasint m = 100, n = 100;
    blasint lda = 100;
    char order = 'R';
    char trans = 'R';

    float alpha[] = {1.0f, 1.0f};
    float beta[] = {1.0f, 1.5f};

    blasint inc_x = 2;
    blasint inc_y = 1;

    float norm = check_cgemv('C', order, trans, m, n, alpha, lda,
                                inc_x, beta, inc_y);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS_ZGEMV);
}

/**
 * Test error function for an invalid param inc_y.
 * Must be positive
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_inc_y)
{
    char order = 'C';
    char trans = 'T';

    blasint m = 1, n = 1;
    blasint lda = 1;

    blasint inc_x = 1;
    blasint inc_y = 0;

    int expected_info = 11;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_y.
 * Must be positive
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_inc_y_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = 1;
    blasint lda = 1;

    blasint inc_x = 1;
    blasint inc_y = 0;

    int expected_info = 11;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_y.
 * Must be positive
 * 
 * Row major
 */
CTEST(cgemv, c_api_xerbla_invalid_inc_y_row_major)
{
    enum CBLAS_ORDER corder = CblasRowMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasNoTrans;

    blasint m = 1, n = 1;
    blasint lda = 1;

    blasint inc_x = 1;
    blasint inc_y = 0;

    int expected_info = 11;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_x.
 * Must be positive
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_inc_x)
{
    char order = 'C';
    char trans = 'T';
    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 0;
    blasint inc_y = 1;
    
    int expected_info = 8;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_x.
 * Must be positive
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_inc_x_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 0;
    blasint inc_y = 1;
    
    int expected_info = 8;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param inc_x.
 * Must be positive
 * 
 * Row major
 */
CTEST(cgemv, c_api_xerbla_invalid_inc_x_row_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 0;
    blasint inc_y = 1;
    
    int expected_info = 8;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param n.
 * Must be positive.
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_n)
{
    char order = 'C';
    char trans = 'T';

    blasint m = 1, n = INVALID;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 3;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param n.
 * Must be positive.
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_n_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = INVALID;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;
    
    int expected_info = 3;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param n.
 * Must be positive.
 * 
 * Row major
 */
CTEST(cgemv, c_api_xerbla_invalid_n_row_major)
{
    enum CBLAS_ORDER corder = CblasRowMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasNoTrans;

    blasint m = INVALID, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 3;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param m.
 * Must be positive.
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_m)
{
    char order = 'C';
    char trans = 'T';

    blasint m = INVALID, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;
    
    int expected_info = 2;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param m.
 * Must be positive.
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_m_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = INVALID, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 2;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param m.
 * Must be positive.
 * 
 * Row major
 */
CTEST(cgemv, c_api_xerbla_invalid_m_row_major)
{
    enum CBLAS_ORDER corder = CblasRowMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasNoTrans;

    blasint m = 1, n = INVALID;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 2;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda.
 * lda must be at least n.
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_lda)
{
    char order = 'C';
    char trans = 'T';

    blasint m = 1, n = 1;
    blasint lda = INVALID;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 6;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda.
 * If matrices are stored using col major layout, 
 * lda must be at least m.
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_lda_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = 1;
    blasint lda = INVALID;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 6;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda.
 * If matrices are stored using col major layout, 
 * lda must be at least n.
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_lda_row_major)
{
    enum CBLAS_ORDER corder = CblasRowMajor;
    enum CBLAS_TRANSPOSE ctrans = CblasNoTrans;

    blasint m = 1, n = 1;
    blasint lda = INVALID;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 6;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param trans.
 * 
 * Column major
 */
CTEST(cgemv, xerbla_invalid_trans)
{
    char order = 'C';
    char trans = 'Z';

    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 1;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param trans.
 * 
 * Column major
 */
CTEST(cgemv, c_api_xerbla_invalid_trans_col_major)
{
    enum CBLAS_ORDER corder = CblasColMajor;
    enum CBLAS_TRANSPOSE ctrans = INVALID;

    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 1;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param trans.
 * 
 * Row major
 */
CTEST(cgemv, c_api_xerbla_invalid_trans_row_major)
{
    enum CBLAS_ORDER corder = CblasRowMajor;
    enum CBLAS_TRANSPOSE ctrans = INVALID;
    
    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 1;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param order. 
 */
CTEST(cgemv, c_api_xerbla_invalid_order_col_major)
{
    enum CBLAS_ORDER corder = INVALID;
    enum CBLAS_TRANSPOSE ctrans = CblasTrans;

    blasint m = 1, n = 1;
    blasint lda = 1;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    int expected_info = 0;
    int passed;

    passed = c_api_check_badargs(corder, ctrans, m, n, lda, inc_x, inc_y, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif