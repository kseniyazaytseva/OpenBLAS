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

#ifdef BUILD_SINGLE

/**
 * Fortran API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, y1_zero)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 0.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0f;
	tr_d2 = 2.0f;
	tr_x1 = 8.0f;
	tr_y1 = 0.0f;

	tr_param[0] = -2.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	BLASFUNC(srotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, d1_negative)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = -1.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0f;
	tr_d2 = 0.0f;
	tr_x1 = 0.0f;
	tr_y1 = 8.0f;

	tr_param[0] = -1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	BLASFUNC(srotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, d1_positive_d2_positive_x1_zero)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 0.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0f;
	tr_d2 = 2.0f;
	tr_x1 = 8.0f;
	tr_y1 = 8.0f;

	tr_param[0] = 1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	BLASFUNC(srotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * Fortran API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, scaled_y_greater_than_scaled_x)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 1.0f;
	te_d2 = tr_d2 = -2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0f;
	tr_d2 = 0.0f;
	tr_x1 = 0.0f;
	tr_y1 = 8.0f;

	tr_param[0] = -1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	BLASFUNC(srotmg)(&te_d1, &te_d2, &te_x1, &te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * C API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, c_api_y1_zero)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 0.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0f;
	tr_d2 = 2.0f;
	tr_x1 = 8.0f;
	tr_y1 = 0.0f;

	tr_param[0] = -2.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	cblas_srotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * C API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, c_api_d1_negative)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = -1.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0f;
	tr_d2 = 0.0f;
	tr_x1 = 0.0f;
	tr_y1 = 8.0f;

	tr_param[0] = -1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	cblas_srotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * C API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, c_api_d1_positive_d2_positive_x1_zero)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 2.0f;
	te_d2 = tr_d2 = 2.0f;
	te_x1 = tr_x1 = 0.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 2.0f;
	tr_d2 = 2.0f;
	tr_x1 = 8.0f;
	tr_y1 = 8.0f;

	tr_param[0] = 1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	cblas_srotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}

/**
 * C API specific test
 * Test srotmg by comparing it against pre-calculated values
 */
CTEST(srotmg, c_api_scaled_y_greater_than_scaled_x)
{
	float te_d1, tr_d1;
	float te_d2, tr_d2;
	float te_x1, tr_x1;
	float te_y1, tr_y1;
	float te_param[5];
	float tr_param[5];
	int i = 0;
	te_d1 = tr_d1 = 1.0f;
	te_d2 = tr_d2 = -2.0f;
	te_x1 = tr_x1 = 8.0f;
	te_y1 = tr_y1 = 8.0f;

	for(i=0; i<5; i++){
	  te_param[i] = tr_param[i] = 0.0f;
	}
	
	//reference values as calculated by netlib blas
	tr_d1 = 0.0f;
	tr_d2 = 0.0f;
	tr_x1 = 0.0f;
	tr_y1 = 8.0f;

	tr_param[0] = -1.0f;
	tr_param[1] = 0.0f;
	tr_param[2] = 0.0f;
	tr_param[3] = 0.0f;
	tr_param[4] = 0.0f;

	//OpenBLAS
	cblas_srotmg(&te_d1, &te_d2, &te_x1, te_y1, te_param);

	ASSERT_DBL_NEAR_TOL(tr_d1, te_d1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_d2, te_d2, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_x1, te_x1, SINGLE_EPS);
	ASSERT_DBL_NEAR_TOL(tr_y1, te_y1, SINGLE_EPS);

	for(i=0; i<5; i++){
		ASSERT_DBL_NEAR_TOL(tr_param[i], te_param[i], SINGLE_EPS);
	}
}
#endif