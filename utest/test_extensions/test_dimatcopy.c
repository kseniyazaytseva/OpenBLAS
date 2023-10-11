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

struct DATA_DIMATCOPY{
    double A_Test[DATASIZE* DATASIZE];
    double A_Verify[DATASIZE* DATASIZE];
};

#ifdef BUILD_DOUBLE
static struct DATA_DIMATCOPY data_dimatcopy;


static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * Find difference between two rectangle matrix
 * return norm of differences
 */
static double matrix_difference(double *a, double *b, blasint cols, blasint rows, blasint ld)
{
    blasint i = 0;
    blasint j = 0;
    blasint inc = 1;
    double norm = 0.0;

    double *a_ptr = a;
    double *b_ptr = b;

    for(i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++) {
            a_ptr[j] -= b_ptr[j];
        }
        norm += cblas_dnrm2(cols, a_ptr, inc);
        
        a_ptr += ld;
        b_ptr += ld;
    }
    return norm/(double)(rows);
}

/**
 * Transpose matrix
 * 
 * param rows specifies number of rows of A
 * param cols specifies number of columns of A
 * param alpha specifies scaling factor for matrix A
 * param a_src - buffer holding input matrix A
 * param lda_src - leading dimension of the matrix A
 * param a_dst - buffer holding output matrix A
 * param lda_dst - leading dimension of output matrix A
 */
static void transpose(blasint rows, blasint cols, double alpha, double *a_src, int lda_src, 
                      double *a_dst, blasint lda_dst)
{
    blasint i, j;
    for (i = 0; i != cols; i++)
    {
        for (j = 0; j != rows; j++)
            a_dst[i*lda_dst+j] = alpha*a_src[j*lda_src+i];
    }
}

/**
 * Copy matrix from source A to destination A
 * 
 * param rows specifies number of rows of A
 * param cols specifies number of columns of A
 * param alpha specifies scaling factor for matrix A
 * param a_src - buffer holding input matrix A
 * param lda_src - leading dimension of the matrix A
 * param a_dst - buffer holding output matrix A
 * param lda_dst - leading dimension of output matrix A
 */
static void copy(blasint rows, blasint cols, double alpha, double *a_src, int lda_src, 
                      double *a_dst, blasint lda_dst)
{
    blasint i, j;
    for (i = 0; i != rows; i++)
    {
        for (j = 0; j != cols; j++)
            a_dst[i*lda_dst+j] = alpha*a_src[i*lda_src+j];
    }
}

/**
 * Comapare results computed by dimatcopy and reference func
 *
 * param api specifies tested api (C or Fortran)
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation 
 * applied to the matrix A
 * param rows specifies number of rows of A
 * param cols specifies number of columns of A
 * param alpha specifies scaling factor for matrix A
 * param lda_src - leading dimension of the matrix A
 * param lda_dst - leading dimension of output matrix A
 * return norm of difference between openblas and reference func
 */
static double check_dimatcopy(char api, char order, char trans, blasint rows, blasint cols, double alpha, 
                             blasint lda_src, blasint lda_dst)
{
    blasint m, n;
    blasint rows_out, cols_out;
    enum CBLAS_ORDER corder;
    enum CBLAS_TRANSPOSE ctrans;

    if (order == 'C') {
        n = rows; m = cols;
    }
    else {
        m = rows; n = cols;
    }

    if(trans == 'T' || trans == 'C') {
        rows_out = n; cols_out = m;
    }
    else {
        rows_out = m; cols_out = n;
    }

    rand_generate(data_dimatcopy.A_Test, lda_src*m);

    if (trans == 'T' || trans == 'C') {
        transpose(m, n, alpha, data_dimatcopy.A_Test, lda_src, data_dimatcopy.A_Verify, lda_dst);
    } 
    else {
        copy(m, n, alpha, data_dimatcopy.A_Test, lda_src, data_dimatcopy.A_Verify, lda_dst);
    }

    if (api == 'F') {
        BLASFUNC(dimatcopy)(&order, &trans, &rows, &cols, &alpha, data_dimatcopy.A_Test, 
                            &lda_src, &lda_dst);
    }
    else {
        if (order == 'C') corder = CblasColMajor;
        if (order == 'R') corder = CblasRowMajor;
        if (trans == 'T') ctrans = CblasTrans;
        if (trans == 'N') ctrans = CblasNoTrans;
        if (trans == 'C') ctrans = CblasConjTrans;
        if (trans == 'R') ctrans = CblasConjNoTrans;
        cblas_dimatcopy(corder, ctrans, rows, cols, alpha, data_dimatcopy.A_Test, 
                    lda_src, lda_dst);
    }

    // Find the differences between output matrix computed by dimatcopy and reference func
    return matrix_difference(data_dimatcopy.A_Test, data_dimatcopy.A_Verify, cols_out, rows_out, lda_dst);
}

/**
 * Check if error function was called with expected function name
 * and param info
 *
 * param order specifies row or column major order
 * param trans specifies op(A), the transposition operation 
 * applied to the matrix A
 * param rows specifies number of rows of A
 * param cols specifies number of columns of A
 * param lda_src - leading dimension of the matrix A
 * param lda_dst - leading dimension of output matrix A
 * param expected_info - expected invalid parameter number
 * return TRUE if everything is ok, otherwise FALSE
 */
static int check_badargs(char order, char trans, blasint rows, blasint cols,
                          blasint lda_src, blasint lda_dst, int expected_info)
{
    double alpha = 1.0;

    set_xerbla("DIMATCOPY", expected_info);

    BLASFUNC(dimatcopy)(&order, &trans, &rows, &cols, &alpha, data_dimatcopy.A_Test, 
                        &lda_src, &lda_dst);

    return check_error();
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, colmajor_trans_col_100_row_100_alpha_one)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, colmajor_notrans_col_100_row_100_alpha_one)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, colmajor_trans_col_100_row_100_alpha_zero)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, colmajor_notrans_col_100_row_100_alpha_zero)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, colmajor_trans_col_50_row_100_alpha_one)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'T';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, colmajor_notrans_col_50_row_100_alpha_one)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, colmajor_trans_col_50_row_100_alpha_zero)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'T';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, colmajor_notrans_col_50_row_100_alpha_zero)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, colmajor_trans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'T';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, colmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_100_alpha_one)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_100_alpha_one)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_100_alpha_zero)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_100_alpha_zero)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Rectangular matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_50_alpha_one)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'R';
    char trans = 'C'; // same as trans for real matrix
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_50_alpha_one)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 1.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_50_alpha_zero)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'R';
    char trans = 'C'; // same as trans for real matrix
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha = 0.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_50_alpha_zero)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 0.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, rowmajor_trans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'R';
    char trans = 'C'; // same as trans for real matrix
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, rowmajor_notrans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, c_api_colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha = 1.0
 */
CTEST(dimatcopy, c_api_colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, c_api_rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test dimatcopy by comparing it against reference
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha = 2.0
 */
CTEST(dimatcopy, c_api_rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha = 2.0;
    double norm;

    norm = check_dimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test error function for an invalid param order.
 * Must be column (C) or row major (R).
 */
CTEST(dimatcopy, xerbla_invalid_order)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'O';
    char trans = 'T';
    int expected_info = 1;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param trans.
 * Must be trans (T/C) or no-trans (N/R).
 */
CTEST(dimatcopy, xerbla_invalid_trans)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'O';
    int expected_info = 2;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param m.
 * Must be positive.
 */
CTEST(dimatcopy, xerbla_invalid_rows)
{
    blasint m = 0, n = 100;
    blasint lda_src = 0, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 3;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param n.
 * Must be positive.
 */
CTEST(dimatcopy, xerbla_invalid_cols)
{
    blasint m = 100, n = 0;
    blasint lda_src = 100, lda_dst = 0;
    char order = 'C';
    char trans = 'T';
    int expected_info = 4;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_src.
 * If matrices are stored using row major layout, 
 * lda_src must be at least n.
 */
CTEST(dimatcopy, xerbla_rowmajor_invalid_lda)
{
    blasint m = 50, n = 100;
    blasint lda_src = 50, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    int expected_info = 7;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_src.
 * If matrices are stored using column major layout,
 * lda_src must be at least m.
 */
CTEST(dimatcopy, xerbla_colmajor_invalid_lda)
{
    blasint m = 100, n = 50;
    blasint lda_src = 50, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    int expected_info = 7;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_dst.
 * If matrices are stored using row major layout and 
 * there is no transposition, lda_dst must be at least n.
 */
CTEST(dimatcopy, xerbla_rowmajor_notrans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'R';
    char trans = 'N';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_dst.
 * If matrices are stored using row major layout and 
 * there is transposition, lda_dst must be at least m.
 */
CTEST(dimatcopy, xerbla_rowmajor_trans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'R';
    char trans = 'T';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_dst.
 * If matrices are stored using column major layout and 
 * there is no transposition, lda_dst must be at least m.
 */
CTEST(dimatcopy, xerbla_colmajor_notrans_invalid_ldb)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'N';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}

/**
 * Test error function for an invalid param lda_dst.
 * If matrices are stored using column major layout and 
 * there is transposition, lda_dst must be at least n.
 */
CTEST(dimatcopy, xerbla_colmajor_trans_invalid_ldb)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'T';
    int expected_info = 9;
    int passed;

    passed = check_badargs(order, trans, m, n, lda_src, lda_dst, expected_info);
    ASSERT_EQUAL(TRUE, passed);
}
#endif