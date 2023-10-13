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
#include <cblas.h>
#include "common.h"

#define DATASIZE 300
#define INCREMENT 2

struct DATA_CTRMV {
	float a_test[DATASIZE * DATASIZE * 2];
	float a_verify[DATASIZE * DATASIZE * 2];
	float x_test[DATASIZE * INCREMENT * 2];
	float x_verify[DATASIZE * INCREMENT * 2];
};

#ifdef BUILD_COMPLEX
static struct DATA_CTRMV data_ctrmv;

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A
 * and comparing it with the non-conjugate ctrmv.
 *
 * param uplo specifies whether A is upper or lower triangular
 * param trans specifies op(A), the transposition (conjugation) operation applied to A
 * param diag specifies whether the matrix A is unit triangular or not.
 * param n - numbers of rows and columns of A
 * param lda - leading dimension of matrix A
 * param incx - increment for the elements of x
 * return norm of difference
 */
static float check_ctrmv(char uplo, char trans, char diag, blasint n, blasint lda, blasint incx)
{
	blasint i;
	float alpha_conj[] = {1.0f, 0.0f}; 
	char trans_verify = trans;

	srand_generate(data_ctrmv.a_test, n * lda * 2);
	srand_generate(data_ctrmv.x_test, n * incx * 2);

	for (i = 0; i < n * lda * 2; i++)
		data_ctrmv.a_verify[i] = data_ctrmv.a_test[i];

	for (i = 0; i < n * incx * 2; i++)
		data_ctrmv.x_verify[i] = data_ctrmv.x_test[i];

	if (trans == 'R'){
		cblas_cimatcopy(CblasColMajor, CblasConjNoTrans, n, n, alpha_conj, data_ctrmv.a_verify, lda, lda);
		trans_verify = 'N';
	}

	BLASFUNC(ctrmv)(&uplo, &trans_verify, &diag, &n, data_ctrmv.a_verify, &lda,
	 				data_ctrmv.x_verify, &incx);

	BLASFUNC(ctrmv)(&uplo, &trans, &diag, &n, data_ctrmv.a_test, &lda,
	 				data_ctrmv.x_test, &incx);

	for (i = 0; i < n * incx * 2; i++)
		data_ctrmv.x_verify[i] -= data_ctrmv.x_test[i];

	return BLASFUNC(scnrm2)(&n, data_ctrmv.x_verify, &incx);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is upper triangular
 * matrix A is not unit triangular
 */
CTEST(ctrmv, conj_notrans_upper_not_unit_triangular)
{
	blasint n = DATASIZE, incx = 1, lda = DATASIZE;
	char uplo = 'U';
	char diag = 'N';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is upper triangular
 * matrix A is unit triangular
 */
CTEST(ctrmv, conj_notrans_upper_unit_triangular)
{
	blasint n = DATASIZE, incx = 1, lda = DATASIZE;
	char uplo = 'U';
	char diag = 'U';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is lower triangular
 * matrix A is not unit triangular
 */
CTEST(ctrmv, conj_notrans_lower_not_triangular)
{
	blasint n = DATASIZE, incx = 1, lda = DATASIZE;
	char uplo = 'L';
	char diag = 'N';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is lower triangular
 * matrix A is unit triangular
 */
CTEST(ctrmv, conj_notrans_lower_unit_triangular)
{
	blasint n = DATASIZE, incx = 1, lda = DATASIZE;
	char uplo = 'L';
	char diag = 'U';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is upper triangular
 * matrix A is not unit triangular
 * vector x stride is 2
 */
CTEST(ctrmv, conj_notrans_upper_not_unit_triangular_incx_2)
{
	blasint n = DATASIZE, incx = 2, lda = DATASIZE;
	char uplo = 'U';
	char diag = 'N';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is upper triangular
 * matrix A is unit triangular
 * vector x stride is 2
 */
CTEST(ctrmv, conj_notrans_upper_unit_triangular_incx_2)
{
	blasint n = DATASIZE, incx = 2, lda = DATASIZE;
	char uplo = 'U';
	char diag = 'U';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is lower triangular
 * matrix A is not unit triangular
 * vector x stride is 2
 */
CTEST(ctrmv, conj_notrans_lower_not_triangular_incx_2)
{
	blasint n = DATASIZE, incx = 2, lda = DATASIZE;
	char uplo = 'L';
	char diag = 'N';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * Test ctrmv with the conjugate and not-transposed matrix A by conjugating matrix A 
 * and comparing it with the non-conjugate ctrmv.
 * Test with the following options:
 *
 * matrix A is conjugate and not-trans
 * matrix A is lower triangular
 * matrix A is unit triangular
 * vector x stride is 2
 */
CTEST(ctrmv, conj_notrans_lower_unit_triangular_incx_2)
{
	blasint n = DATASIZE, incx = 2, lda = DATASIZE;
	char uplo = 'L';
	char diag = 'U';
	char trans = 'R';

	float norm = check_ctrmv(uplo, trans, diag, n, lda, incx);

	ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif