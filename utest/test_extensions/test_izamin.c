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
CTEST(izamin, bad_args_N_0){
   blasint i;
   blasint N = 0, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < ELEMENTS * inc * 2; i ++) {
      x[i] = 1000 - i;
   }
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(izamin, step_zero){
   blasint i;
   blasint N = ELEMENTS, inc = 0;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * 2; i ++) {
      x[i] = i - 1000;
   }
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(0, index);
}

CTEST(izamin, positive_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {1.0, 2.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(izamin, negative_step_1_N_1){
   blasint N = 1, inc = 1;
   double x[] = {-1.0, -2.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(izamin, positive_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(izamin, negative_step_2_N_1){
   blasint N = 1, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(1, index);
}

CTEST(izamin, positive_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_1_N_2){
   blasint N = 2, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_2_N_2){
   blasint N = 2, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_1_N_3){
   blasint N = 3, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -2.0, -1.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_2_N_3){
   blasint N = 3, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0, -2.0, -2.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_1_N_4){
   blasint N = 4, inc = 1;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -2.0, -1.0, -2.0, -2.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, negative_step_2_N_4){
   blasint N = 4, inc = 2;
   double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(2, index);
}

CTEST(izamin, positive_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[7 * inc * 2] = 0.0;
   x[7 * inc * 2 + 1] = 0.0;
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(8, index);
}

CTEST(izamin, negative_step_1_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = 1;
   double x[ELEMENTS * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = - i - 1000;
   }
   x[7 * inc * 2] = 0.0;
   x[7 * inc * 2 + 1] = 0.0;
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(8, index);
}

CTEST(izamin, positive_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = i + 1000;
   }
   x[7 * inc * 2] = 0.0;
   x[7 * inc * 2 + 1] = 0.0;
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(8, index);
}

CTEST(izamin, negative_step_2_N_50){
   blasint i;
   blasint N = ELEMENTS, inc = INCREMENT;
   double x[ELEMENTS * INCREMENT * 2];
   for (i = 0; i < N * inc * 2; i ++) {
      x[i] = - i - 1000;
   }
   x[7 * inc * 2] = 0.0;
   x[7 * inc * 2 + 1] = 0.0;
   blasint index = BLASFUNC(izamin)(&N, x, &inc);
   ASSERT_EQUAL(8, index);
}

CTEST(izamin, c_api_bad_args_N_0){
    blasint i;
    blasint N = 0, inc = 1;
    double x[ELEMENTS * 2];
    for (i = 0; i < ELEMENTS * inc * 2; i ++) {
        x[i] = 1000 - i;
    }
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_step_zero){
    blasint i;
    blasint N = ELEMENTS, inc = 0;
    double x[ELEMENTS * 2];
    for (i = 0; i < N * 2; i ++) {
        x[i] = i - 1000;
    }
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_positive_step_1_N_1){
    blasint N = 1, inc = 1;
    double x[] = {1.0, 2.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_negative_step_1_N_1){
    blasint N = 1, inc = 1;
    double x[] = {-1.0, -2.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_positive_step_2_N_1){
    blasint N = 1, inc = 2;
    double x[] = {1.0, 2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_negative_step_2_N_1){
    blasint N = 1, inc = 2;
    double x[] = {-1.0, -2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(0, index);
}

CTEST(izamin, c_api_positive_step_1_N_2){
    blasint N = 2, inc = 1;
    double x[] = {1.0, 2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_1_N_2){
    blasint N = 2, inc = 1;
    double x[] = {-1.0, -2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_2_N_2){
    blasint N = 2, inc = 2;
    double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_2_N_2){
    blasint N = 2, inc = 2;
    double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_1_N_3){
    blasint N = 3, inc = 1;
    double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_1_N_3){
    blasint N = 3, inc = 1;
    double x[] = {-1.0, -2.0, 0.0, 0.0, -2.0, -1.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_2_N_3){
    blasint N = 3, inc = 2;
    double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_2_N_3){
    blasint N = 3, inc = 2;
    double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_1_N_4){
    blasint N = 4, inc = 1;
    double x[] = {1.0, 2.0, 0.0, 0.0, 2.0, 1.0, -2.0, -2.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_1_N_4){
    blasint N = 4, inc = 1;
    double x[] = {-1.0, -2.0, 0.0, 0.0, -2.0, -1.0, -2.0, -2.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_2_N_4){
    blasint N = 4, inc = 2;
    double x[] = {1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_negative_step_2_N_4){
    blasint N = 4, inc = 2;
    double x[] = {-1.0, -2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0};
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(1, index);
}

CTEST(izamin, c_api_positive_step_1_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = 1;
    double x[ELEMENTS * 2];
    for (i = 0; i < N * inc * 2; i ++) {
        x[i] = i + 1000;
    }
    x[7 * inc * 2] = 0.0;
    x[7 * inc * 2 + 1] = 0.0;
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(7, index);
}

CTEST(izamin, c_api_negative_step_1_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = 1;
    double x[ELEMENTS * 2];
    for (i = 0; i < N * inc * 2; i ++) {
        x[i] = - i - 1000;
    }
    x[7 * inc * 2] = 0.0;
    x[7 * inc * 2 + 1] = 0.0;
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(7, index);
}

CTEST(izamin, c_api_positive_step_2_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = INCREMENT;
    double x[ELEMENTS * INCREMENT * 2];
    for (i = 0; i < N * inc * 2; i ++) {
        x[i] = i + 1000;
    }
    x[7 * inc * 2] = 0.0;
    x[7 * inc * 2 + 1] = 0.0;
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(7, index);
}

CTEST(izamin, c_api_negative_step_2_N_50){
    blasint i;
    blasint N = ELEMENTS, inc = INCREMENT;
    double x[ELEMENTS * INCREMENT * 2];
    for (i = 0; i < N * inc * 2; i ++) {
        x[i] = - i - 1000;
    }
    x[7 * inc * 2] = 0.0;
    x[7 * inc * 2 + 1] = 0.0;
    blasint index = cblas_izamin(N, x, inc);
    ASSERT_EQUAL(7, index);
}
#endif
