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

#define ELEMENTS 50
#define INCREMENT 2

#ifdef BUILD_COMPLEX

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   float x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0f;
   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {1.1f, -1.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.1f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {1.1f, 0.0f, 2.3f, -1.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {1.1f, -1.0f, 2.3f, -1.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.4f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {1.1f, -1.5f, 1.1f, -1.0f, 1.0f, 1.0f, 1.1f, -1.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.6f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 1.1f, -1.0f, 0.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.4f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {1.1f, 0.0f, -1.0f, 0.0f, -1.0f, -3.0f, -1.0f, 0.0f, 2.2f, 3.0f, -1.0f, 0.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2.3f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {1.1f, 1.0f, -2.2f, 3.3f, 1.1f, 1.0f, -2.2f, 3.3f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(6.4f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {1.1f, 0.0f, 1.1f, 1.0f, 1.0f, 2.0f, 1.1f, 1.0f, 2.2f, 2.7f, 1.1f, 1.0f, -3.3f, -5.9f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(-0.2f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {0.0f, 1.0f, 2.2f, 3.3f, 0.0f, 0.0f, 1.0f, 2.2f, 3.3f, 0.0f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(13.0f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {0.0f, 3.0f, 1.0f, 2.2f, 1.0f, -2.2f, 1.0f, 2.2f, 2.2f, -1.7f, 1.0f, 2.2f, 3.3f, 14.5f, 1.0f, 2.2f, 0.0f, -9.0f, 1.0f, 2.2f};

   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(11.1f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0f : 1.0f;
   }
   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * Fortran API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0f : 1.0f;
   }
   float sum = BLASFUNC(scsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   float x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0f;
   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {1.1f, -1.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.1f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {1.1f, 0.0f, 2.3f, -1.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.1f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {1.1f, -1.0f, 2.3f, -1.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.4f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {1.1f, -1.5f, 1.1f, -1.0f, 1.0f, 1.0f, 1.1f, -1.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.6f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 1.1f, -1.0f, 0.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(4.4f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {1.1f, 0.0f, -1.0f, 0.0f, -1.0f, -3.0f, -1.0f, 0.0f, 2.2f, 3.0f, -1.0f, 0.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(2.3f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {1.1f, 1.0f, -2.2f, 3.3f, 1.1f, 1.0f, -2.2f, 3.3f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(6.4f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {1.1f, 0.0f, 1.1f, 1.0f, 1.0f, 2.0f, 1.1f, 1.0f, 2.2f, 2.7f, 1.1f, 1.0f, -3.3f, -5.9f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(-0.2f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {0.0f, 1.0f, 2.2f, 3.3f, 0.0f, 0.0f, 1.0f, 2.2f, 3.3f, 0.0f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(13.0f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {0.0f, 3.0f, 1.0f, 2.2f, 1.0f, -2.2f, 1.0f, 2.2f, 2.2f, -1.7f, 1.0f, 2.2f, 3.3f, 14.5f, 1.0f, 2.2f, 0.0f, -9.0f, 1.0f, 2.2f};

   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(11.1f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0f : 1.0f;
   }
   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}

/**
 * C API specific test
 * Test scsum by comparing it against pre-calculated values
 */
CTEST(scsum, c_api_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0f : 1.0f;
   }
   float sum = cblas_scsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0f, sum, SINGLE_EPS);
}
#endif