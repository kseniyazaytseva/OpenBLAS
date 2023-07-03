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

#ifdef BUILD_COMPLEX16
CTEST(dzsum, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1, -1.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0, 2.3, -1.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, -1.0, 2.3, -1.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, -1.5, 1.1, -1.0, 1.0, 1.0, 1.1, -1.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.6, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 1.1, -1.0, 0.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, -1.0, 0.0, -1.0, -3.0, -1.0, 0.0, 2.2, 3.0, -1.0, 0.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2.3, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, -2.2, 3.3, 1.1, 1.0, -2.2, 3.3};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(6.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.1, 1.0, 1.0, 2.0, 1.1, 1.0, 2.2, 2.7, 1.1, 1.0, -3.3, -5.9};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(-0.2, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {0.0, 1.0, 2.2, 3.3, 0.0, 0.0, 1.0, 2.2, 3.3, 0.0};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(13.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {0.0, 3.0, 1.0, 2.2, 1.0, -2.2, 1.0, 2.2, 2.2, -1.7, 1.0, 2.2, 3.3, 14.5, 1.0, 2.2, 0.0, -9.0, 1.0, 2.2};

   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(11.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = BLASFUNC(dzsum)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1, -1.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0, 2.3, -1.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, -1.0, 2.3, -1.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, -1.5, 1.1, -1.0, 1.0, 1.0, 1.1, -1.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(1.6, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 1.1, -1.0, 0.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(4.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, -1.0, 0.0, -1.0, -3.0, -1.0, 0.0, 2.2, 3.0, -1.0, 0.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(2.3, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, -2.2, 3.3, 1.1, 1.0, -2.2, 3.3};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(6.4, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.1, 1.0, 1.0, 2.0, 1.1, 1.0, 2.2, 2.7, 1.1, 1.0, -3.3, -5.9};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(-0.2, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {0.0, 1.0, 2.2, 3.3, 0.0, 0.0, 1.0, 2.2, 3.3, 0.0};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(13.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {0.0, 3.0, 1.0, 2.2, 1.0, -2.2, 1.0, 2.2, 2.2, -1.7, 1.0, 2.2, 3.3, 14.5, 1.0, 2.2, 0.0, -9.0, 1.0, 2.2};

   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(11.1, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}

CTEST(dzsum, c_api_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = (i & 1) ? -1.0 : 1.0;
   }
   double sum = cblas_dzsum(N, x, inc);
   ASSERT_DBL_NEAR_TOL(0.0, sum, DOUBLE_EPS);
}
#endif