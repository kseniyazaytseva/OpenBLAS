/***************************************************************************
Copyright (c) 2020, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"
#if !defined(DOUBLE)
#define VSETVL(n) vsetvl_e32m4(n)
#define VSETVL_MAX vsetvlmax_e32m1()
#define FLOAT_V_T vfloat32m4_t
#define FLOAT_V_T_M1 vfloat32m1_t
#define VLEV_FLOAT vle32_v_f32m4
#define VLSEV_FLOAT vlse32_v_f32m4
#define VSEV_FLOAT vse32_v_f32m4
#define VSSEV_FLOAT vsse32_v_f32m4
#define VFREDSUM_FLOAT vfredusum_vs_f32m4_f32m1
#define VFMACCVV_FLOAT vfmacc_vv_f32m4
#define VFMACCVF_FLOAT vfmacc_vf_f32m4
#define VFMVVF_FLOAT vfmv_v_f_f32m4
#define VFMVVF_FLOAT_M1 vfmv_v_f_f32m1
#define VFMULVV_FLOAT vfmul_vv_f32m4
#else
#define VSETVL(n) vsetvl_e64m4(n)
#define VSETVL_MAX vsetvlmax_e64m1()
#define FLOAT_V_T vfloat64m4_t
#define FLOAT_V_T_M1 vfloat64m1_t
#define VLEV_FLOAT vle64_v_f64m4
#define VLSEV_FLOAT vlse64_v_f64m4
#define VSEV_FLOAT vse64_v_f64m4
#define VSSEV_FLOAT vsse64_v_f64m4
#define VFREDSUM_FLOAT vfredusum_vs_f64m4_f64m1
#define VFMACCVV_FLOAT vfmacc_vv_f64m4
#define VFMACCVF_FLOAT vfmacc_vf_f64m4
#define VFMVVF_FLOAT vfmv_v_f_f64m4
#define VFMVVF_FLOAT_M1 vfmv_v_f_f64m1
#define VFMULVV_FLOAT vfmul_vv_f64m4
#endif

int CNAME(BLASLONG m, BLASLONG offset, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y, FLOAT *buffer)
{
        BLASLONG i, j, k;
        BLASLONG ix,iy;
        BLASLONG jx,jy;
        FLOAT temp1;
        FLOAT temp2;
        FLOAT *a_ptr = a;
        unsigned int gvl = 0;
        FLOAT_V_T_M1 v_res, v_z0;
        gvl = VSETVL_MAX;
        v_res = VFMVVF_FLOAT_M1(0, gvl);
        v_z0 = VFMVVF_FLOAT_M1(0, gvl);

        FLOAT_V_T va, vx, vy, vr;
        BLASLONG stride_x, stride_y, inc_xv, inc_yv, len;

        if(inc_x == 1 && inc_y == 1){
                for (j=0; j<offset; j++)
                {
                        temp1 = alpha * x[j];
                        temp2 = 0.0;
                        y[j] += temp1 * a_ptr[j];
                        i = j + 1;
                        len = m - i;
                        if(len > 0){
                                gvl = VSETVL(len);
                                vr = VFMVVF_FLOAT(0, gvl);
                                for(k = 0; k < len / gvl; k++){
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VLEV_FLOAT(&y[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSEV_FLOAT(&y[i], vy, gvl);

                                        vx = VLEV_FLOAT(&x[i], gvl);
                                        vr = VFMACCVV_FLOAT(vr, vx, va, gvl);

                                        i += gvl;
                                }
                                v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                temp2 = EXTRACT_FLOAT(v_res);
                                if(i < m){
                                        gvl = VSETVL(m-i);
                                        vy = VLEV_FLOAT(&y[i], gvl);
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSEV_FLOAT(&y[i], vy, gvl);

                                        vx = VLEV_FLOAT(&x[i], gvl);
                                        vr = VFMULVV_FLOAT(vx, va, gvl);
                                        v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                        temp2 += EXTRACT_FLOAT(v_res);
                                }
			}
                        y[j] += alpha * temp2;
                        a_ptr += lda;
                }
        }else if(inc_x == 1){
                jy = 0;
                stride_y = inc_y * sizeof(FLOAT);
                for (j=0; j<offset; j++)
                {
                        temp1 = alpha * x[j];
                        temp2 = 0.0;
                        y[jy] += temp1 * a_ptr[j];
                        iy = jy + inc_y;
                        i = j + 1;
                        len = m - i;
                        if(len > 0){
                                gvl = VSETVL(len);
                                inc_yv = inc_y * gvl;
                                vr = VFMVVF_FLOAT(0, gvl);
                                for(k = 0; k < len / gvl; k++){
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VLSEV_FLOAT(&y[iy], stride_y, gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSSEV_FLOAT(&y[iy], stride_y, vy, gvl);

                                        vx = VLEV_FLOAT(&x[i], gvl);
                                        vr = VFMACCVV_FLOAT(vr, vx, va, gvl);

                                        i += gvl;
                                        iy += inc_yv;
                                }
                                v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                temp2 = EXTRACT_FLOAT(v_res);
                                if(i < m){
                                        gvl = VSETVL(m-i);
                                        vy = VLSEV_FLOAT(&y[iy], stride_y, gvl);
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSSEV_FLOAT(&y[iy], stride_y, vy, gvl);

                                        vx = VLEV_FLOAT(&x[i], gvl);
                                        vr = VFMULVV_FLOAT(vx, va, gvl);
                                        v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                        temp2 += EXTRACT_FLOAT(v_res);
                                }
			}
                        y[jy] += alpha * temp2;
                        jy    += inc_y;
                        a_ptr += lda;
                }
        }else if(inc_y == 1){
                jx = 0;
                stride_x = inc_x * sizeof(FLOAT);
                for (j=0; j<offset; j++)
                {
                        temp1 = alpha * x[jx];
                        temp2 = 0.0;
                        y[j] += temp1 * a_ptr[j];
                        ix = jx + inc_x;
                        i = j + 1;
                        len = m - i;
                        if(len > 0){
                                gvl = VSETVL(len);
                                vr = VFMVVF_FLOAT(0, gvl);
                                inc_xv = inc_x * gvl;
                                for(k = 0; k < len / gvl; k++){
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VLEV_FLOAT(&y[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSEV_FLOAT(&y[i], vy, gvl);

                                        vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                                        vr = VFMACCVV_FLOAT(vr, vx, va, gvl);

                                        i += gvl;
                                        ix += inc_xv;
                                }
                                v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                temp2 = EXTRACT_FLOAT(v_res);
                                if(i < m){
                                        gvl = VSETVL(m-i);
                                        vy = VLEV_FLOAT(&y[i], gvl);
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSEV_FLOAT(&y[i], vy, gvl);

                                        vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                                        vr = VFMULVV_FLOAT(vx, va, gvl);
                                        v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                        temp2 += EXTRACT_FLOAT(v_res);
                                }
			}
                        y[j] += alpha * temp2;
                        jx    += inc_x;
                        a_ptr += lda;
                }
        }else{
                stride_x = inc_x * sizeof(FLOAT);
                stride_y = inc_y * sizeof(FLOAT);
                jx = 0;
                jy = 0;
                for (j=0; j<offset; j++)
                {
                        temp1 = alpha * x[jx];
                        temp2 = 0.0;
                        y[jy] += temp1 * a_ptr[j];
                        ix = jx + inc_x;
                        iy = jy + inc_y;
                        i = j + 1;
                        len = m - i;
                        if(len > 0){
                                gvl = VSETVL(len);
                                inc_xv = inc_x * gvl;
                                inc_yv = inc_y * gvl;
                                vr = VFMVVF_FLOAT(0, gvl);
                                for(k = 0; k < len / gvl; k++){
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VLSEV_FLOAT(&y[iy], stride_y, gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSSEV_FLOAT(&y[iy], stride_y, vy, gvl);

                                        vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                                        vr = VFMACCVV_FLOAT(vr, vx, va, gvl);

                                        i += gvl;
                                        ix += inc_xv;
                                        iy += inc_yv;
                                }
                                v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                temp2 = EXTRACT_FLOAT(v_res);
                                if(i < m){
                                        gvl = VSETVL(m-i);
                                        vy = VLSEV_FLOAT(&y[iy], stride_y, gvl);
                                        va = VLEV_FLOAT(&a_ptr[i], gvl);
                                        vy = VFMACCVF_FLOAT(vy, temp1, va, gvl);
                                        VSSEV_FLOAT(&y[iy], stride_y, vy, gvl);

                                        vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                                        vr = VFMULVV_FLOAT(vx, va, gvl);
                                        v_res = VFREDSUM_FLOAT(v_res, vr, v_z0, gvl);
                                        temp2 += EXTRACT_FLOAT(v_res);
                                }
			}
                        y[jy] += alpha * temp2;
                        jx    += inc_x;
                        jy    += inc_y;
                        a_ptr += lda;
                }
        }
        return(0);
}

