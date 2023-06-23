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
CTEST(idamin, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, positive_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, negative_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {-1.1};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, positive_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, negative_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {-1.1, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, positive_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, 1.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {-1.1, 1.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {-1.1, 1.0, -2.2};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 3.3};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {-1.1, 1.0, -2.2, -3.3};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 3.3, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, negative_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0, -3.3, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(idamin, positive_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 3.3, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(idamin, negative_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {-1.1, 1.0, -2.2, -3.3, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(idamin, positive_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 3.3, 0.0, 0.0, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(idamin, negative_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0, -3.3, 0.0, 0.0, 0.0};

   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(5, index);
}

CTEST(idamin, positive_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0;
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(idamin, negative_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0;
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(idamin, positive_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0;
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(idamin, negative_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0;
   blasint index = BLASFUNC(idamin)(&N, x, &inc);
   ASSERT_EQUAL(9, index);
}

CTEST(idamin, c_api_bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < ELEMENTS * inc; i ++) {
      x[i] = 1000 - i;
   }
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = i + 1000;
   }
   x[8] = 0.0;
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_positive_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.1};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_negative_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {-1.1};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_positive_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.1, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_negative_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {-1.1, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(0, index);
}

CTEST(idamin, c_api_positive_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.1, 1.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {-1.1, 1.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.1, 1.0, 2.2};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {-1.1, 1.0, -2.2};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 3.3};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {-1.1, 1.0, -2.2, -3.3};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 3.3, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_negative_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0, -3.3, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(1, index);
}

CTEST(idamin, c_api_positive_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {1.1, 1.0, 2.2, 3.3, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(idamin, c_api_negative_step_1_N_5){
   blasint N = 5, inc = 1;
   double x[] = {-1.1, 1.0, -2.2, -3.3, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(idamin, c_api_positive_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 3.3, 0.0, 0.0, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(idamin, c_api_negative_step_2_N_5){
   blasint N = 5, inc = 2;
   double x[] = {-1.1, 0.0, 1.0, 0.0, -2.2, 0.0, -3.3, 0.0, 0.0, 0.0};

   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(4, index);
}

CTEST(idamin, c_api_positive_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0;
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}

CTEST(idamin, c_api_negative_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS];
   for (i = 0; i < N  * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0;
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}

CTEST(idamin, c_api_positive_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = i + 1000;
   }

   x[8 * inc] = 0.0;
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}

CTEST(idamin, c_api_negative_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT];
   for (i = 0; i < N * inc; i ++) {
      x[i] = - i - 1000;
   }

   x[8 * inc] = -1.0;
   blasint index = cblas_idamin(N, x, inc);
   ASSERT_EQUAL(8, index);
}
#endif