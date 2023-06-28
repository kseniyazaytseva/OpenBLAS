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

#include "openblas_utest.h"
#include <cblas.h>

#define ELEMENTS 50
#define INCREMENT 2

#ifdef BUILD_DOUBLE
CTEST(dsum, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, -1.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.1, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, -1.5, 1.0, 1.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2.1, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.3, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, -1.0, -3.0, 2.2, 3.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2.3, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, -2.2, 3.3};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.2, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 2.0, 2.2, 2.7, -3.3, -5.9};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {0.0, 1.0, 2.2, 3.3, 0.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(6.5, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {0.0, 3.0, 1.0, -2.2, 2.2, -1.7, 3.3, 14.5, 0.0, -9.0};

   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(6.5, sum, DOUBLE_EPS);
}

CTEST(dsum, step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = BLASFUNC(dsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(50.0, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, -1.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.1, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, -1.5, 1.0, 1.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(2.1, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(4.3, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, -1.0, -3.0, 2.2, 3.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(2.3, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, -2.2, 3.3};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(3.2, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 2.0, 2.2, 2.7, -3.3, -5.9};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.0, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {0.0, 1.0, 2.2, 3.3, 0.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(6.5, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {0.0, 3.0, 1.0, -2.2, 2.2, -1.7, 3.3, 14.5, 0.0, -9.0};

   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(6.5, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dsum, c_api_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = cblas_dsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(50.0, sum, DOUBLE_EPS);
}
#endif