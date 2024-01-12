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

#define LMUL m8
#if defined(DOUBLE)
#       define ELEN 64
#       define ABS fabs
#else
#       define ELEN 32
#       define ABS fabsf
#endif

#define _
#define JOIN2_X(x, y) x ## y
#define JOIN2(x, y) JOIN2_X(x, y)
#define JOIN(v, w, x, y, z) JOIN2( JOIN2( JOIN2( JOIN2( v, w ), x), y), z)

#define VSETVL          JOIN(vsetvl,    _e,     ELEN,   LMUL,   _)
#define FLOAT_V_T       JOIN(vfloat,    ELEN,   LMUL,   _t,     _)
#define FLOAT_V_T_M1    JOIN(vfloat,    ELEN,   m1,     _t,     _)
#define VLEV_FLOAT      JOIN(vle,       ELEN,   _v_f,   ELEN,   LMUL)
#define VLSEV_FLOAT     JOIN(vlse,      ELEN,   _v_f,   ELEN,   LMUL)
#define VFREDMINVS_FLOAT JOIN(vfredmin_vs_f,  ELEN,   LMUL,   _f, JOIN2( ELEN,   m1))
#define VFABS_FLOAT     JOIN(vfabs,      _v_f,  ELEN,   LMUL,   _)
#define VFMVVF_FLOAT    JOIN(vfmv,      _v_f_f  ELEN,   LMUL,   _)
#define VFMVVF_FLOAT_M1 JOIN(vfmv,      _v_f_f, ELEN,   m1,     _)

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x)
{
        BLASLONG i=0, j=0;
        BLASLONG ix=0;
        FLOAT minf=0.0;
        if (n <= 0 || inc_x <= 0) return(minf);

        minf = ABS(*x);
        x += inc_x;
        --n;
        if (n == 0) return(minf);

        unsigned int gvl = 0;
        FLOAT_V_T v0, v1;
        FLOAT_V_T_M1 v_res;
        v_res = VFMVVF_FLOAT_M1(minf, 1);

        if(inc_x == 1){
                gvl = VSETVL(n);
                if(gvl <= n/2){
                        for(i=0,j=0; i<n/(gvl*2); i++){
                                v0 = VLEV_FLOAT(&x[j], gvl);
                                v1 = VLEV_FLOAT(&x[j+gvl], gvl);
                                v0 = VFABS_FLOAT(v0, gvl);
                                v1 = VFABS_FLOAT(v1, gvl);
                                v_res = VFREDMINVS_FLOAT(v_res, v0, v_res, gvl);
                                v_res = VFREDMINVS_FLOAT(v_res, v1, v_res, gvl);
                                j += gvl*2;
                        }
                }
                for(;j<n;){
                        gvl = VSETVL(n-j);
                        v0 = VLEV_FLOAT(&x[j], gvl);
                        v0 = VFABS_FLOAT(v0, gvl);
                        v_res = VFREDMINVS_FLOAT(v_res, v0, v_res, gvl);
                        j += gvl;
                }
        }else{
                gvl = VSETVL(n);
                BLASLONG stride_x = inc_x * sizeof(FLOAT);
                if(gvl <= n/2){
                        BLASLONG inc_xv = inc_x * gvl;
                        for(i=0,j=0; i<n/(gvl*2); i++){
                                v0 = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                                v1 = VLSEV_FLOAT(&x[ix+inc_xv], stride_x, gvl);
                                v0 = VFABS_FLOAT(v0, gvl);
                                v1 = VFABS_FLOAT(v1, gvl);
                                v_res = VFREDMINVS_FLOAT(v_res, v0, v_res, gvl);
                                v_res = VFREDMINVS_FLOAT(v_res, v1, v_res, gvl);
                                j += gvl*2;
                                ix += inc_xv*2;
                        }
                }
                for(;j<n;){
                        gvl = VSETVL(n-j);
                        v0 = VLSEV_FLOAT(&x[j*inc_x], stride_x, gvl);
                        v0 = VFABS_FLOAT(v0, gvl);
                        v_res = VFREDMINVS_FLOAT(v_res, v0, v_res, gvl);
                        j += gvl;
                }
        }

        minf = EXTRACT_FLOAT(v_res);
        return(minf);
}
