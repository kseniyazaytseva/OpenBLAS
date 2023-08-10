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

struct DATA_ZIMATCOPY{
    double A_Test[DATASIZE * DATASIZE * 2];
    double A_Verify[DATASIZE * DATASIZE * 2];
};

#ifdef BUILD_COMPLEX16
static struct DATA_ZIMATCOPY data_zimatcopy;


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

    // Сomplex multiplier
    ld *= 2;

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
 * param conj specifies conjugation
 */
static void transpose(blasint rows, blasint cols, double *alpha, double *a_src, int lda_src, 
                      double *a_dst, blasint lda_dst, int conj)
{
    blasint i, j;
    lda_dst *= 2;
    lda_src *= 2;
    for (i = 0; i != cols*2; i+=2)
    {
        for (j = 0; j != rows*2; j+=2){
            a_dst[(i/2)*lda_dst+j] = alpha[0] * a_src[(j/2)*lda_src+i] + conj * alpha[1] * a_src[(j/2)*lda_src+i+1];
            a_dst[(i/2)*lda_dst+j+1] = (-1.0) * conj * alpha[0] * a_src[(j/2)*lda_src+i+1] + alpha[1] * a_src[(j/2)*lda_src+i];
        } 
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
 * param conj specifies conjugation
 */
static void copy(blasint rows, blasint cols, double *alpha, double *a_src, int lda_src, 
                      double *a_dst, blasint lda_dst, int conj)
{
    blasint i, j;
    lda_dst *= 2;
    lda_src *= 2;
    for (i = 0; i != rows; i++)
    {
        for (j = 0; j != cols*2; j+=2){
            a_dst[i*lda_dst+j] = alpha[0] * a_src[i*lda_src+j] + conj * alpha[1] * a_src[i*lda_src+j+1];
            a_dst[i*lda_dst+j+1] = (-1.0) * conj *alpha[0] * a_src[i*lda_src+j+1] + alpha[1] * a_src[i*lda_src+j];
        }
    }
}

/**
 * Comapare results computed by zimatcopy and reference func
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
static double check_zimatcopy(char api, char order, char trans, blasint rows, blasint cols, double *alpha, 
                             blasint lda_src, blasint lda_dst)
{
    blasint m, n;
    blasint rows_out, cols_out;
    enum CBLAS_ORDER corder;
    enum CBLAS_TRANSPOSE ctrans;
    int conj = -1;

    if (order == 'C') {
        n = rows; m = cols;
    }
    else {
        m = rows; n = cols;
    }

    if(trans == 'T' || trans == 'C') {
        rows_out = n; cols_out = m*2;
        if (trans == 'C')
            conj = 1;
    }
    else {
        rows_out = m; cols_out = n*2;
        if (trans == 'R')
            conj = 1;
    }

    rand_generate(data_zimatcopy.A_Test, lda_src*m*2);

    if (trans == 'T' || trans == 'C') {
        transpose(m, n, alpha, data_zimatcopy.A_Test, lda_src, data_zimatcopy.A_Verify, lda_dst, conj);
    } 
    else {
        copy(m, n, alpha, data_zimatcopy.A_Test, lda_src, data_zimatcopy.A_Verify, lda_dst, conj);
    }

    if (api == 'F') {
        BLASFUNC(zimatcopy)(&order, &trans, &rows, &cols, alpha, data_zimatcopy.A_Test, 
                            &lda_src, &lda_dst);
    }
    else {
        if (order == 'C') corder = CblasColMajor;
        if (order == 'R') corder = CblasRowMajor;
        if (trans == 'T') ctrans = CblasTrans;
        if (trans == 'N') ctrans = CblasNoTrans;
        if (trans == 'C') ctrans = CblasConjTrans;
        if (trans == 'R') ctrans = CblasConjNoTrans;
        cblas_zimatcopy(corder, ctrans, rows, cols, alpha, data_zimatcopy.A_Test, 
                    lda_src, lda_dst);
    }

    // Find the differences between output matrix computed by zimatcopy and reference func
    return matrix_difference(data_zimatcopy.A_Test, data_zimatcopy.A_Verify, cols_out, rows_out, lda_dst);    
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
    double alpha[] = {1.0, 1.0};

    set_xerbla("ZIMATCOPY", expected_info);

    BLASFUNC(zimatcopy)(&order, &trans, &rows, &cols, alpha, data_zimatcopy.A_Test, 
                        &lda_src, &lda_dst);

    return check_error();
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha_r = -3.0, alpha_i = 1.0
 */
CTEST(zimatcopy, colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha[] = {-3.0, 1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, colmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'R';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(zimatcopy, colmajor_conjtrnas_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'C';
    double alpha[] = {2.0, 1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, colmajor_trans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 50;
    char order = 'C';
    char trans = 'T';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Rectangular matrix
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(zimatcopy, colmajor_notrans_col_50_row_100)
{
    blasint m = 100, n = 50;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha[] = {2.0, 1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Rectangular matrix
 * alpha_r = 0.0, alpha_i = 0.0
 */
CTEST(zimatcopy, colmajor_trans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda_src = 50, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha[] = {0.0, 0.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha_r = 2.0, alpha_i = 3.0
 */
CTEST(zimatcopy, rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha[] = {2.0, 3.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Matrix dimensions leave residues from 4 and 2 (specialize 
 * for rt case)
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(zimatcopy, rowmajor_trans_col_27_row_27)
{
    blasint m = 27, n = 27;
    blasint lda_src = 27, lda_dst = 27;
    char order = 'R';
    char trans = 'T'; 
    double alpha[] = {2.0, 1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Rectangular matrix
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(zimatcopy, rowmajor_notrans_col_100_row_50)
{
    blasint m = 50, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N'; 
    double alpha[] = {2.0, 1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy and conjugate
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(zimatcopy, rowmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'R'; 
    double alpha[] = {1.5, -1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, rowmajor_conjtrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'C';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition
 * Square matrix
 * alpha_r = 3.0, alpha_i = 2.0
 */
CTEST(zimatcopy, c_api_colmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'T';
    double alpha[] = {3.0, 2.0};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy only
 * Square matrix
 * alpha_r = 3.0, alpha_i = 1.5
 */
CTEST(zimatcopy, c_api_colmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'N';
    double alpha[] = {3.0, 1.5};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition
 * Square matrix
 * alpha_r = 3.0, alpha_i = 1.0
 */
CTEST(zimatcopy, c_api_rowmajor_trans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'T';
    double alpha[] = {3.0, 1.0};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Copy and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, c_api_colmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'R';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Column Major
 * Transposition and conjugate
 * alpha_r = 2.0, alpha_i = 1.0
 */
CTEST(zimatcopy, c_api_colmajor_conjtrnas_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'C';
    char trans = 'C';
    double alpha[] = {2.0, 1.0};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy only
 * Square matrix
 * alpha_r = 1.0, alpha_i = 1.0
 */
CTEST(zimatcopy, c_api_rowmajor_notrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'N';
    double alpha[] = {1.0, 1.0};

    double norm = check_zimatcopy('C', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Copy and conjugate
 * alpha_r = 1.5, alpha_i = -1.0
 */
CTEST(zimatcopy, c_api_rowmajor_conj_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'R'; 
    double alpha[] = {1.5, -1.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test zimatcopy by comparing it against refernce
 * with the following options:
 *
 * Row Major
 * Transposition and conjugate
 * alpha_r = 1.0, alpha_i = 2.0
 */
CTEST(zimatcopy, c_api_rowmajor_conjtrans_col_100_row_100)
{
    blasint m = 100, n = 100;
    blasint lda_src = 100, lda_dst = 100;
    char order = 'R';
    char trans = 'C';
    double alpha[] = {1.0, 2.0};

    double norm = check_zimatcopy('F', order, trans, m, n, alpha, lda_src, lda_dst);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * Test error function for an invalid param order.
 * Must be column (C) or row major (R).
 */
CTEST(zimatcopy, xerbla_invalid_order)
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
CTEST(zimatcopy, xerbla_invalid_trans)
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
CTEST(zimatcopy, xerbla_invalid_rows)
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
CTEST(zimatcopy, xerbla_invalid_cols)
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
CTEST(zimatcopy, xerbla_rowmajor_invalid_lda)
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
CTEST(zimatcopy, xerbla_colmajor_invalid_lda)
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
CTEST(zimatcopy, xerbla_rowmajor_notrans_invalid_ldb)
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
CTEST(zimatcopy, xerbla_rowmajor_trans_invalid_ldb)
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
CTEST(zimatcopy, xerbla_colmajor_notrans_invalid_ldb)
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
CTEST(zimatcopy, xerbla_colmajor_trans_invalid_ldb)
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