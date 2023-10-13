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

struct DATA_SOMATCOPY {
    float a_test[DATASIZE * DATASIZE];
    float b_test[DATASIZE * DATASIZE];
    float b_verify[DATASIZE * DATASIZE];
};

#ifdef BUILD_SINGLE
static struct DATA_SOMATCOPY data_somatcopy;

/**
 * Comapare results computed by somatcopy and reference func
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
static float check_somatcopy(char api, char order, char trans, blasint rows, blasint cols, float alpha, 
                             blasint lda, blasint ldb)
{
    blasint b_rows, b_cols;
    blasint m, n;
    enum CBLAS_ORDER corder;
    enum CBLAS_TRANSPOSE ctrans;

    if (order == 'C') {
        m = cols; n = rows;
    }
    else {
        m = rows; n = cols;
    }

    if(trans == 'T' || trans == 'C') {
        b_rows = n; b_cols = m;
    }
    else {
        b_rows = m; b_cols = n;
    }

    srand_generate(data_somatcopy.a_test, lda*m);

    if (trans == 'T' || trans == 'C') {
        stranspose(m, n, alpha, data_somatcopy.a_test, lda, data_somatcopy.b_verify, ldb);
    } 
    else {
        scopy(m, n, alpha, data_somatcopy.a_test, lda, data_somatcopy.b_verify, ldb);
    }

    if (api == 'F') {
        BLASFUNC(somatcopy)(&order, &trans, &rows, &cols, &alpha, data_somatcopy.a_test, 
                            &lda, data_somatcopy.b_test, &ldb);
    }
    else {
        if (order == 'C') corder = CblasColMajor;
        if (order == 'R') corder = CblasRowMajor;
        if (trans == 'T') ctrans = CblasTrans;
        if (trans == 'N') ctrans = CblasNoTrans;
        if (trans == 'C') ctrans = CblasConjTrans;
        if (trans == 'R') ctrans = CblasConjNoTrans;
        cblas_somatcopy(corder, ctrans, rows, cols, alpha, data_somatcopy.a_test, 
                    lda, data_somatcopy.b_test, ldb);
    }
    
    return smatrix_difference(data_somatcopy.b_test, data_somatcopy.b_verify, b_cols, b_rows, ldb);
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
    float alpha = 1.0;

    set_xerbla("SOMATCOPY", expected_info);

    BLASFUNC(somatcopy)(&order, &trans, &rows, &cols, &alpha, data_somatcopy.a_test, 
                        &lda, data_somatcopy.b_test, &ldb);

    return check_error();
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'T';
    float alpha = 1.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'N';
    float alpha = 1.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(somatcopy, colmajor_trans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'T';
    float alpha = 2.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(somatcopy, colmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'N';
    float alpha = 2.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(somatcopy, colmajor_trans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 50, ldb = 100;
    char order = 'C';
    char trans = 'T';
    float alpha = 0.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(somatcopy, colmajor_notrans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 50, ldb = 50;
    char order = 'C';
    char trans = 'N';
    float alpha = 0.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'T';
    float alpha = 1.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'N';
    float alpha = 1.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(somatcopy, rowmajor_conjtrans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'C'; // same as trans for real matrix
    float alpha = 2.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(somatcopy, rowmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda = 50, ldb = 50;
    char order = 'R';
    char trans = 'N'; 
    float alpha = 2.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Matrix dimensions leave residues from 4 and 2 (specialize
 * for rt case)
 * alpha = 1.5
 */
CTEST(somatcopy, rowmajor_trans_col_27_row_27)
{
    blasint m = 27, n = 27;
    blasint lda = 27, ldb = 27;
    char order = 'R';
    char trans = 'T'; 
    float alpha = 1.5f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(somatcopy, rowmajor_notrans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'N'; 
    float alpha = 0.0f;

    float norm = check_somatcopy('F', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, c_api_colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'T';
    float alpha = 1.0f;

    float norm = check_somatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, c_api_colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'N';
    float alpha = 1.0f;

    float norm = check_somatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, c_api_rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'T';
    float alpha = 1.0f;

    float norm = check_somatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * Test somatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(somatcopy, c_api_rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'R';
    char trans = 'N';
    float alpha = 1.0f;

    float norm = check_somatcopy('C', order, trans, m, n, alpha, lda, ldb);

    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test error function for an invalid param order.
 * Must be column (C) or row major (R).
 */
CTEST(somatcopy, xerbla_invalid_order)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'O';
    char trans = 'T';
    int expected_info = 1;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param trans.
 * Must be trans (T/C) or no-trans (N/R).
 */
CTEST(somatcopy, xerbla_invalid_trans)
{
    blasint m = 100, n = 100;
    blasint lda = 100, ldb = 100;
    char order = 'C';
    char trans = 'O';
    int expected_info = 2;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param m.
 * Must be positive.
 */
CTEST(somatcopy, xerbla_invalid_rows)
{
    blasint m = 0, n = 100;
    blasint lda = 0, ldb = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 3;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param n.
 * Must be positive.
 */
CTEST(somatcopy, xerbla_invalid_cols)
{
    blasint m = 100, n = 0;
    blasint lda = 100, ldb = 0;
    char order = 'C';
    char trans = 'T';
    int expected_info = 4;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda.
 * If matrices are stored using row major layout,
 * lda must be at least n.
 */
CTEST(somatcopy, xerbla_rowmajor_invalid_lda)
{
    blasint m = 50, n = 100;
    blasint lda = 50, ldb = 100;
    char order = 'R';
    char trans = 'T';
    int expected_info = 7;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda.
 * If matrices are stored using column major layout,
 * lda must be at least m.
 */
CTEST(somatcopy, xerbla_colmajor_invalid_lda)
{
    blasint m = 100, n = 50;
    blasint lda = 50, ldb = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 7;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param ldb.
 * If matrices are stored using row major layout and
 * there is no transposition, ldb must be at least n.
 */
CTEST(somatcopy, xerbla_rowmajor_notrans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'N';
    int expected_info = 9;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param ldb.
 * If matrices are stored using row major layout and
 * there is transposition, ldb must be at least m.
 */
CTEST(somatcopy, xerbla_rowmajor_trans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'R';
    char trans = 'T';
    int expected_info = 9;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param ldb.
 * If matrices are stored using column major layout and
 * there is no transposition, ldb must be at least m.
 */
CTEST(somatcopy, xerbla_colmajor_notrans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'N';
    int expected_info = 9;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param ldb.
 * If matrices are stored using column major layout and
 * there is transposition, ldb must be at least n.
 */
CTEST(somatcopy, xerbla_colmajor_trans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda = 100, ldb = 50;
    char order = 'C';
    char trans = 'T';
    int expected_info = 9;

    int passed = check_badargs(order, trans, m, n, lda, ldb, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif
