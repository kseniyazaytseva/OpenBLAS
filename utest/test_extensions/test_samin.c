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

#ifdef BUILD_SINGLE
CTEST(samin, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0f;
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {1.1f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {-1.1f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {1.1f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {-1.1f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.1f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {1.1f, 1.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {-1.1f, 1.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f, 0.0f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f, 0.0f, 0.0f};

   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_1_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0f;
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(0.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_1_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0f;
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, positive_step_2_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 1.0f;
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}

CTEST(samin, negative_step_2_N_70){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0f;
   float amin = BLASFUNC(samin)(&N, x, &inc);
   ASSERT_DBL_NEAR_TOL(1.0f, amin, SINGLE_EPS);
}
#endif