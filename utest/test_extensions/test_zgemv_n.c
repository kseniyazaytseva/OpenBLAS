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

struct DATA_ZSPMV{
    double A_test[DATASIZE * DATASIZE * 2];
    double B_test[DATASIZE * 2 * INCREMENT];
    double C_test[DATASIZE * 2 * INCREMENT];
    double C_verify[DATASIZE * 2 * INCREMENT];
};

#ifdef BUILD_COMPLEX16

static struct DATA_ZSPMV data_zgemv;

static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * zgemv not transposed reference code
 *
 * param trans specifies whether matris A is conj or/and xconj
 * param m - number of rows of A
 * param n - number of columns of A
 * param alpha - scaling factor for the matrib-vector product
 * param a - buffer holding input matrib A
 * param lda - leading dimension of matrix A
 * param b - Buffer holding input vector b
 * param inc_b - stride of vector b
 * param beta - scaling factor for vector c
 * param c - buffer holding input/output vector c
 * param inc_c - stride of vector c
 */
static void zgemv_n_trusted(char trans, blasint m, blasint n, double *alpha, double *a,
                          blasint lda, double *b, blasint inc_b, double *beta, double *c,
                          blasint inc_c)
{
	blasint i, j;
    blasint i2 = 0;
	blasint ib = 0, ic = 0;

    double temp_r, temp_i;

	double *a_ptr = a;
    blasint lda2 = 2*lda;

	blasint inc_b2 = 2 * inc_b;
    blasint inc_c2 = 2 * inc_c;

    BLASFUNC(zscal)(&m, beta, c, &inc_c);

	for (j = 0; j < n; j++)
	{

        if (trans == 'N' || trans == 'R') {
            temp_r = alpha[0] * b[ib] - alpha[1] * b[ib+1];
            temp_i = alpha[0] * b[ib+1] + alpha[1] * b[ib];
        } else {
            temp_r = alpha[0] * b[ib] + alpha[1] * b[ib+1];
            temp_i = alpha[0] * b[ib+1] - alpha[1] * b[ib];
        }

		ic = 0;
		i2 = 0;

		for (i = 0; i < m; i++)
		{
                if (trans == 'N') {
                    c[ic] += temp_r * a_ptr[i2] - temp_i * a_ptr[i2+1];
                    c[ic+1] += temp_r * a_ptr[i2+1] + temp_i * a_ptr[i2];
                } 
                if (trans == 'O') {
                    c[ic] += temp_r * a_ptr[i2] + temp_i * a_ptr[i2+1];
                    c[ic+1] += temp_r * a_ptr[i2+1] - temp_i * a_ptr[i2];
                }
                if (trans == 'R') {
                    c[ic] += temp_r * a_ptr[i2] + temp_i * a_ptr[i2+1];
                    c[ic+1] -= temp_r * a_ptr[i2+1] - temp_i * a_ptr[i2];
                }
                if (trans == 'S') {
                    c[ic] += temp_r * a_ptr[i2] - temp_i * a_ptr[i2+1];
                    c[ic+1] -= temp_r * a_ptr[i2+1] + temp_i * a_ptr[i2];
                }
			i2 += 2;
			ic += inc_c2;
		}
		a_ptr += lda2;
		ib += inc_b2;
	}

}

/**
 * Comapare results computed by zgemv and zgemv_n_trusted
 *
 * param trans specifies whether matris A is conj or/and xconj
 * param m - number of rows of A
 * param n - number of columns of A
 * param alpha - scaling factor for the matrib-vector product
 * param lda - leading dimension of matrix A
 * param inc_b - stride of vector b
 * param beta - scaling factor for vector c
 * param inc_c - stride of vector c
 * return norm of differences
 */
static double check_zgemv_n(char trans, blasint m, blasint n, double *alpha, blasint lda, 
                            blasint inc_b, double *beta, blasint inc_c)
{
    blasint i;

    rand_generate(data_zgemv.A_test, n * lda);
    rand_generate(data_zgemv.B_test, 2 * n * inc_b);
    rand_generate(data_zgemv.C_test, 2 * m * inc_c);

    for (i = 0; i < m * 2 * inc_c; i++)
        data_zgemv.C_verify[i] = data_zgemv.C_test[i];

    zgemv_n_trusted(trans, m, n, alpha, data_zgemv.A_test, lda, data_zgemv.B_test, 
                  inc_b, beta, data_zgemv.C_test, inc_c);
    BLASFUNC(zgemv)(&trans, &m, &n, alpha, data_zgemv.A_test, &lda, data_zgemv.B_test, 
                    &inc_b, beta, data_zgemv.C_verify, &inc_c);

    for (i = 0; i < m * 2 * inc_c; i++)
        data_zgemv.C_verify[i] -= data_zgemv.C_test[i];

    return BLASFUNC(dznrm2)(&n, data_zgemv.C_verify, &inc_c);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_o_square_matrix)
{
    blasint n = 100, m = 100, lda = 100;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'O';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj
 * Number of rows of A is 50
 * Number of colums of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_o_rectangular_matrix_rows_less_then_cols)
{
    blasint n = 100, m = 50, lda = 50;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'O';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj
 * Number of rows of A is 100
 * Number of colums of A is 50
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_o_rectangular_matrix_cols_less_then_rows)
{
    blasint n = 50, m = 100, lda = 100;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'O';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 2
 */
CTEST(zgemv, trans_o_double_strides)
{
    blasint n = 100, m = 100, lda = 100;
    blasint inc_b = 2, inc_c = 2;
    char trans = 'O';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj and conj
 * Number of rows and columns of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_s_square_matrix)
{
    blasint n = 100, m = 100, lda = 100;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'S';
    double alpha[] = {1.0, 1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj and conj
 * Number of rows of A is 50
 * Number of colums of A is 100
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_s_rectangular_matrix_rows_less_then_cols)
{
    blasint n = 100, m = 50, lda = 50;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'S';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj and conj
 * Number of rows of A is 100
 * Number of colums of A is 50
 * Stride of vector b is 1
 * Stride of vector c is 1
 */
CTEST(zgemv, trans_s_rectangular_matrix_cols_less_then_rows)
{
    blasint n = 50, m = 100, lda = 100;
    blasint inc_b = 1, inc_c = 1;
    char trans = 'S';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.4, 0.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

/**
 * Test zgemv by comparing it against reference
 * with the following options:
 *
 * A is xconj and conj
 * Number of rows and columns of A is 100
 * Stride of vector b is 2
 * Stride of vector c is 2
 */
CTEST(zgemv, trans_s_double_strides)
{
    blasint n = 100, m = 100, lda = 100;
    blasint inc_b = 2, inc_c = 2;
    char trans = 'S';
    double alpha[] = {2.0, -1.0};
    double beta[] = {1.0, 5.0};
    double norm = 0.0;

    norm = check_zgemv_n(trans, m, n, alpha, lda, inc_b, beta, inc_c);

    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_TOL);
}

#endif
