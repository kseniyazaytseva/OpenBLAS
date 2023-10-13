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

#ifndef _TEST_EXTENSION_COMMON_H_
#define _TEST_EXTENSION_COMMON_H_

#include <cblas.h>
#include <ctype.h>

#define TRUE 1
#define FALSE 0
#define INVALID -1
#define SINGLE_TOL 1e-02f
#define DOUBLE_TOL 1e-10

extern int check_error(void);
extern void set_xerbla(char* current_rout, int expected_info);
extern int BLASFUNC(xerbla)(char *name, blasint *info, blasint length);

extern void srand_generate(float *alpha, blasint n);
extern void drand_generate(double *alpha, blasint n);

extern float smatrix_difference(float *a, float *b, blasint cols, blasint rows, blasint ld);
extern double dmatrix_difference(double *a, double *b, blasint cols, blasint rows, blasint ld);

extern void cconjugate_vector(blasint n, blasint inc_x, float *x_ptr);
extern void zconjugate_vector(blasint n, blasint inc_x, double *x_ptr);

extern void stranspose(blasint rows, blasint cols, float alpha, float *a_src, int lda_src, 
                       float *a_dst, blasint lda_dst);
extern void dtranspose(blasint rows, blasint cols, double alpha, double *a_src, int lda_src, 
                double *a_dst, blasint lda_dst);
extern void ctranspose(blasint rows, blasint cols, float *alpha, float *a_src, int lda_src, 
                      float *a_dst, blasint lda_dst, int conj);
extern void ztranspose(blasint rows, blasint cols, double *alpha, double *a_src, int lda_src, 
                double *a_dst, blasint lda_dst, int conj);

extern void scopy(blasint rows, blasint cols, float alpha, float *a_src, int lda_src, 
           float *a_dst, blasint lda_dst);
extern void dcopy(blasint rows, blasint cols, double alpha, double *a_src, int lda_src, 
           double *a_dst, blasint lda_dst);
extern void ccopy(blasint rows, blasint cols, float *alpha, float *a_src, int lda_src, 
           float *a_dst, blasint lda_dst, int conj);
extern void zcopy(blasint rows, blasint cols, double *alpha, double *a_src, int lda_src, 
           double *a_dst, blasint lda_dst, int conj);                
#endif