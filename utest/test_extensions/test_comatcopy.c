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

struct DATA_COMATCOPY{
    float A_test[DATASIZE * DATASIZE * 2];
    float B_test[DATASIZE * DATASIZE * 2];
    float b_verify[DATASIZE * DATASIZE * 2];
};

#ifdef BUILD_COMPLEX
static struct DATA_COMATCOPY data_comatcopy;

/**
 * Transpose complex matrix out-of-place
 *
 * param rows specifies number of rows of A and number of columns of B
 * param cols specifies number of columns of A and number of rows of B
 * param alpha specifies scaling factor for matrix B
 * param a - buffer holding input matrix A
 * param lda - leading dimension of the matrix A
 * param b - buffer holding output matrix B
 * param ldb - leading dimension of the matrix B
 * param conj specifies conjugation
 */
static void transpose(blasint rows, blasint cols, float* alpha, float* a, int lda, 
                      float* b, blasint ldb, int conj)
{
    blasint i, j;
    ldb *= 2;
    lda *= 2;
    for (i = 0; i != cols*2; i+=2)
    {
        for (j = 0; j != rows*2; j+=2){
            b[(i/2)*ldb+j] = alpha[0]*a[(j/2)*lda+i] + conj * alpha[1] * a[(j/2)*lda+i+1];
            b[(i/2)*ldb+j+1] = (-1.0f)*conj*alpha[0]*a[(j/2)*lda+i+1] + alpha[1] * a[(j/2)*lda+i];
        } 
    }
}

/**
 * Copy complex matrix from A to B
 *
 * param rows specifies number of rows of A and B
 * param cols specifies number of columns of A and B
 * param alpha specifies scaling factor for matrix B
 * param a - buffer holding input matrix A
 * param lda - leading dimension of the matrix A
 * param b - buffer holding output matrix B
 * param ldb - leading dimension of the matrix B
 * param conj specifies conjugation
 */
static void copy(blasint rows, blasint cols, float* alpha, float* a, int lda, 
                      float* b, blasint ldb, int conj)
{
    blasint i, j;
    ldb *= 2;
    lda *= 2;
    for (i = 0; i != rows; i++)
    {
        for (j = 0; j != cols*2; j+=2){
            b[i*ldb+j] = alpha[0] * a[i*lda+j] + conj * alpha[1] * a[i*lda+j+1];
            b[i*ldb+j+1] = (-1.0f) * conj *alpha[0] * a[i*lda+j+1] + alpha[1] * a[i*lda+j];
        }
    }
}

static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
 * Comapare results computed by comatcopy and reference func
 *
 * param api specifies tested api (C or Fortran)
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation
 * applied to the matrix A
 * param rows - number of rows of A
 * param cols - number of columns of A
 * param alpha - scaling factor for matrix B
 * param lda - leading dimension of the matrix A
 * param ldb - leading dimension of the matrix B
 * return norm of difference between openblas and reference func
 */
static float check_comatcopy(char api, char order, char trans, blasint rows, blasint cols, float* alpha, 
                             blasint lda, blasint ldb)
{
    blasint i, j;
    blasint b_rows, b_cols;
    blasint m, n;
    blasint inc = 1;
    enum CBLAS_ORDER corder;
    enum CBLAS_TRANSPOSE ctrans;
    float norm = 0.0f;
    int conj = -1;

    if (order == 'C') {
        m = cols; n = rows;
    }
    else {
        m = rows; n = cols;
    }

    if(trans == 'T' || trans == 'C') {
        b_rows = n; b_cols = m*2;
        if (trans == 'C')
            conj = 1;
    }
    else {
        b_rows = m; b_cols = n*2;
        if (trans == 'R')
            conj = 1;
    }

    rand_generate(data_comatcopy.A_test, lda*m*2);

    if (trans == 'T' || trans == 'C') {
        transpose(m, n, alpha, data_comatcopy.A_test, lda, data_comatcopy.b_verify, ldb, conj);
    } 
    else {
        copy(m, n, alpha, data_comatcopy.A_test, lda, data_comatcopy.b_verify, ldb, conj);
    }

    if (api == 'F') {
        BLASFUNC(comatcopy)(&order, &trans, &rows, &cols, alpha, data_comatcopy.A_test, 
                            &lda, data_comatcopy.B_test, &ldb);
    }
    else {
        if (order == 'C') corder = CblasColMajor;
        if (order == 'R') corder = CblasRowMajor;
        if (trans == 'T') ctrans = CblasTrans;
        if (trans == 'N') ctrans = CblasNoTrans;
        if (trans == 'C') ctrans = CblasConjTrans;
        if (trans == 'R') ctrans = CblasConjNoTrans;
        cblas_comatcopy(corder, ctrans, rows, cols, alpha, data_comatcopy.A_test, 
                    lda, data_comatcopy.B_test, ldb);
    }

    for(i = 0; i < b_rows; i++)
    {
        for (j = 0; j < b_cols; j++)
            data_comatcopy.B_test[i*ldb*2+j] -= data_comatcopy.b_verify[i*ldb*2+j];

        norm += BLASFUNC(snrm2)(&b_cols, data_comatcopy.B_test+ldb*2*i, &inc);
    }
    
    return norm/(float)(b_rows);
}

/**
 * Check if error function was called with expected function name
 * and param info
 *
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation
 * applied to the matrix A
 * param rows - number of rows of A
 * param cols - number of columns of A
 * param lda - leading dimension of the matrix A
 * param ldb - leading dimension of the matrix B
 * param expected_info - expected invalid parameter number
 * return TRUE if everything is ok, otherwise FALSE
 */
static int check_badargs(char order, char trans, blasint rows, blasint cols,
                          blasint lda, blasint ldb, int expected_info)
{
    float alpha[] = {1.0f, 1.0f};

    set_xerbla("COMATCOPY", expected_info);

    BLASFUNC(comatcopy)(&order, &trans, &rows, &cols, alpha, data_comatcopy.A_test, 
                        &lda, data_comatcopy.B_test, &ldb);

    return check_error();
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, colmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'N';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * alpha_r = -1.0, alpha_i = 2.0
 */
CTEST(comatcopy, colmajor_trans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'T';
    float alpha[] = {-1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, colmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'R';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(comatcopy, colmajor_conjtrnas_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'C';
    float alpha[] = {2.0f, 1.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, rowmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 50, ldb = 50;
    char order = 'R';
    char trans = 'N'; 
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, rowmajor_trans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'T';
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy and conjugate
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, rowmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'R'; 
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, rowmajor_conjtrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'C';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, c_api_colmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'N';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * alpha_r = -1.0, alpha_i = 2.0
 */
CTEST(comatcopy, c_api_colmajor_trans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'T';
    float alpha[] = {-1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, c_api_colmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'R';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(comatcopy, c_api_colmajor_conjtrnas_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'C';
    float alpha[] = {2.0f, 1.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, c_api_rowmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 50, ldb = 50;
    char order = 'R';
    char trans = 'N'; 
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, c_api_rowmajor_trans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'T';
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy and conjugate
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(comatcopy, c_api_rowmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'R'; 
    float alpha[] = {1.5f, -1.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test comatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(comatcopy, c_api_rowmajor_conjtrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'C';
    float alpha[] = {1.0f, 2.0f};
    float norm;

    norm = check_comatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
* Test error function for an invalid param order.
* Must be column (C) or row major (R).
*/
CTEST(comatcopy, xerbla_invalid_order)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'O';
    char trans = 'T';
    int expected_info = 1;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param trans.
* Must be trans (T/C) or no-trans (N/R).
*/
CTEST(comatcopy, xerbla_invalid_trans)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'O';
    int expected_info = 2;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param m.
* Must be positive.
*/
CTEST(comatcopy, xerbla_invalid_rows)
{
    blasint m = 0, n = 100;
    blasint lda = 0, ldb = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 3;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param n.
* Must be positive.
*/
CTEST(comatcopy, xerbla_invalid_cols)
{
    blasint m = 100, n = 0;
    blasint lda = 100, ldb = 0;
    char order = 'C';
    char trans = 'T';
    int expected_info = 4;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param lda.
* If matrices are stored using row major layout, 
* lda must be at least n.
*/
CTEST(comatcopy, xerbla_rowmajor_invalid_lda)
{
    blasint m = 50, n = 100;
    blasint lda = 50, ldb = 100;
    char order = 'R';
    char trans = 'T';
    int expected_info = 7;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param lda.
* If matrices are stored using column major layout,
* lda must be at least m.
*/
CTEST(comatcopy, xerbla_colmajor_invalid_lda)
{
    blasint m = 100, n = 50;
    blasint lda = 50, ldb = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 7;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using row major layout and 
* there is no transposition, ldb must be at least n.
*/
CTEST(comatcopy, xerbla_rowmajor_notrans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'N';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using row major layout and 
* there is transposition, ldb must be at least m.
*/
CTEST(comatcopy, xerbla_rowmajor_trans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'T';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using row major layout and 
* there is no transposition, ldb must be at least n.
*/
CTEST(comatcopy, xerbla_rowmajor_conj_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'R';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using row major layout and 
* there is transposition, ldb must be at least m.
*/
CTEST(comatcopy, xerbla_rowmajor_transconj_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'C';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using column major layout and 
* there is no transposition, ldb must be at least m.
*/
CTEST(comatcopy, xerbla_colmajor_notrans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'N';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using column major layout and 
* there is transposition, ldb must be at least n.
*/
CTEST(comatcopy, xerbla_colmajor_trans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'T';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using column major layout and 
* there is no transposition, ldb must be at least m.
*/
CTEST(comatcopy, xerbla_colmajor_conj_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'R';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
* Test error function for an invalid param ldb.
* If matrices are stored using column major layout and 
* there is transposition, ldb must be at least n.
*/
CTEST(comatcopy, xerbla_colmajor_transconj_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'C';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif
