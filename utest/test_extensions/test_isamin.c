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

#ifdef BUILD_SINGLE
CTEST(isamin, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, positive_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {1.1f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, negative_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {-1.1f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, positive_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {1.1f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, negative_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {-1.1f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, positive_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {1.1f, 1.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {-1.1f, 1.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, negative_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(isamin, positive_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(isamin, negative_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(isamin, positive_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f, 0.0f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(isamin, negative_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f, 0.0f, 0.0f};

   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(isamin, positive_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(isamin, negative_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(isamin, positive_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(isamin, negative_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(isamin, min_idx_in_vec_tail){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[(N - 1) * inc] = 0.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(N, index);
}

CTEST(isamin, min_idx_in_vec_tail_inc_1){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS * inc];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[(N - 1) * inc] = 0.0f;
   blasint index = BLASFUNC(isamin)(&N, x, &inc);
   ASSERT_EQUAL(N, index);
}

CTEST(isamin, c_api_bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   float x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   float x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0f;
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_positive_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {1.1f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_negative_step_1_N_1){
   blasint N = 1, inc = 1;
   float x[] = {-1.1f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_positive_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {1.1f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_negative_step_2_N_1){
   blasint N = 1, inc = 2;
   float x[] = {-1.1f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(isamin, c_api_positive_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {1.1f, 1.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_1_N_2){
   blasint N = 2, inc = 1;
   float x[] = {-1.1f, 1.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_2_N_2){
   blasint N = 2, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_1_N_3){
   blasint N = 3, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_2_N_3){
   blasint N = 3, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_1_N_4){
   blasint N = 4, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_negative_step_2_N_4){
   blasint N = 4, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(isamin, c_api_positive_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {1.1f, 1.0f, 2.2f, 3.3f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(isamin, c_api_negative_step_1_N_5){
   blasint N = 5, inc = 1;
   float x[] = {-1.1f, 1.0f, -2.2f, -3.3f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(isamin, c_api_positive_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 3.3f, 0.0f, 0.0f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(isamin, c_api_negative_step_2_N_5){
   blasint N = 5, inc = 2;
   float x[] = {-1.1f, 0.0f, 1.0f, 0.0f, -2.2f, 0.0f, -3.3f, 0.0f, 0.0f, 0.0f};

   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(isamin, c_api_positive_step_1_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = 1;
    float x[ELEMENTS];
    for (i = 0; i < N * inc; i ++) {
        x[i] = i + 1000;
    }

    x[8 * inc] = 0.0f;
    blasint index = cblas_isamin(N, x, inc);
    ASSERT_EQUAL(8, index);
}

CTEST(isamin, c_api_negative_step_1_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = 1;
    float x[ELEMENTS];
    for (i = 0; i < N  * inc; i ++) {
        x[i] = - i - 1000;
    }

    x[8 * inc] = -1.0f;
    blasint index = cblas_isamin(N, x, inc);
    ASSERT_EQUAL(8, index);
}

CTEST(isamin, c_api_positive_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0f;
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}

CTEST(isamin, c_api_negative_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0f;
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}

CTEST(isamin, c_api_min_idx_in_vec_tail){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   float x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[(N - 1) * inc] = 0.0f;
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(N - 1, index);
}

CTEST(isamin, c_api_min_idx_in_vec_tail_inc_1){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   float x[ELEMENTS * inc];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[(N - 1) * inc] = 0.0f;
   blasint index = cblas_isamin(N, x, inc);
   ASSERT_EQUAL(N - 1, index);
}
#endif