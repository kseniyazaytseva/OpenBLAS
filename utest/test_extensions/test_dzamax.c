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

#define ELEMENTS 70
#define INCREMENT 2

#ifdef BUILD_COMPLEX16
CTEST(dzamax, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, step_zero){
   blasint i;
   blasint N = ELEMENTS * 2, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.0, 2.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {-1.0, -2.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(3.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -3.0, -1.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -3.0, -1.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0, -2.0, -2.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -2.0, -1.0, -2.0, -2.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(4.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_1_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = i;
   }
   x[7 * inc * 2] = 1000.0;
   x[7 * inc * 2 + 1] = 1000.0;
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2000.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_1_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = -i;
   }
   x[7 * inc * 2] = 1000.0;
   x[7 * inc * 2 + 1] = 1000.0;
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2000.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, positive_step_2_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = i;
   }
   x[7 * inc * 2] = 1000.0;
   x[7 * inc * 2 + 1] = 1000.0;
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2000.0, amax, DOUBLE_EPS);
}

CTEST(dzamax, negative_step_2_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = -i;
   }
   x[7 * inc * 2] = 1000.0;
   x[7 * inc * 2 + 1] = 1000.0;
   double amax = BLASFUNC(dzamax)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(2000.0, amax, DOUBLE_EPS);
}
#endif