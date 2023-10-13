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

#ifdef BUILD_DOUBLE

/**
 * Fortran API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, y1_zero)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 0.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0;
	tr_d2 = 2.0;
	tr_x1 = 8.0;
	tr_y1 = 0.0;

	tr_param[0] = -2.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, d1_negative)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = -1.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0;
	tr_d2 = 0.0;
	tr_x1 = 0.0;
	tr_y1 = 8.0;

	tr_param[0] = -1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, d1_positive_d2_positive_x1_zero)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 0.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0;
	tr_d2 = 2.0;
	tr_x1 = 8.0;
	tr_y1 = 8.0;

	tr_param[0] = 1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, scaled_y_greater_than_scaled_x)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 1.0;
	te_d2 = tr_d2 = -2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0;
	tr_d2 = 0.0;
	tr_x1 = 0.0;
	tr_y1 = 8.0;

	tr_param[0] = -1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	BLASFUNC(drotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * C API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, c_api_y1_zero)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 0.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0;
	tr_d2 = 2.0;
	tr_x1 = 8.0;
	tr_y1 = 0.0;

	tr_param[0] = -2.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	cblas_drotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * C API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, c_api_d1_negative)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = -1.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0;
	tr_d2 = 0.0;
	tr_x1 = 0.0;
	tr_y1 = 8.0;

	tr_param[0] = -1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	cblas_drotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * C API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, c_api_d1_positive_d2_positive_x1_zero)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0;
	te_d2 = tr_d2 = 2.0;
	te_x1 = tr_x1 = 0.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0;
	tr_d2 = 2.0;
	tr_x1 = 8.0;
	tr_y1 = 8.0;

	tr_param[0] = 1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	cblas_drotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}

/**
 * C API specific test
 * Test drotmg by comparing it against pre-calculated values
 */
CTEST(drotmg, c_api_scaled_y_greater_than_scaled_x)
{
	double te_d1, tr_d1;
	double te_d2, tr_d2;
	double te_x1, tr_x1;
	double te_y1, tr_y1;
	double te_param[5];
	double tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 1.0;
	te_d2 = tr_d2 = -2.0;
	te_x1 = tr_x1 = 8.0;
	te_y1 = tr_y1 = 8.0;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0;
	tr_d2 = 0.0;
	tr_x1 = 0.0;
	tr_y1 = 8.0;

	tr_param[0] = -1.0;
	tr_param[1] = 0.0;
	tr_param[2] = 0.0;
	tr_param[3] = 0.0;
	tr_param[4] = 0.0;

	//OpenBLAS
	cblas_drotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, DOUBLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, DOUBLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], DOUBLE_EPS);
	}
}
#endif