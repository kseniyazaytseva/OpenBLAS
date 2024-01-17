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
#include <math.h>

#ifdef RISCV64_ZVL256B
#       define LMUL m2
#       if defined(DOUBLE)
#               define ELEN 64
#               define MLEN _b32
#       else
#               define ELEN 32
#               define MLEN _b16
#       endif
#else
#       define LMUL m8
#       if defined(DOUBLE)
#               define ELEN 64
#               define MLEN _b8
#       else
#               define ELEN 32
#               define MLEN _b4
#       endif
#endif

#define _
#define JOIN2_X(x, y) x ## y
#define JOIN2(x, y) JOIN2_X(x, y)
#define JOIN(v, w, x, y, z) JOIN2( JOIN2( JOIN2( JOIN2( v, w ), x), y), z)

#define VSETVL          JOIN(__riscv_vsetvl,    _e,     ELEN,   LMUL,   _)
#define FLOAT_V_T       JOIN(vfloat,            ELEN,   LMUL,   _t,     _)
#define FLOAT_V_T_M1    JOIN(vfloat,            ELEN,   m1,     _t,     _)
#define VLEV_FLOAT      JOIN(__riscv_vle,       ELEN,   _v_f,   ELEN,   LMUL)
#define VLSEV_FLOAT     JOIN(__riscv_vlse,      ELEN,   _v_f,   ELEN,   LMUL)
#define VFREDSUMVS_FLOAT JOIN(__riscv_vfredusum_vs_f,  ELEN,   LMUL,   _f, JOIN2( ELEN,   m1))
#define VFMVVF_FLOAT    JOIN(__riscv_vfmv,      _v_f_f, ELEN,   LMUL,   _)
#define VFMVVF_FLOAT_M1 JOIN(__riscv_vfmv,      _v_f_f, ELEN,   m1,     _)
#define VFADDVV_FLOAT   JOIN(__riscv_vfadd,     _vv_f,  ELEN,   LMUL,   _)
#define VMFLTVF_FLOAT   JOIN(__riscv_vmflt,     _vf_f,  ELEN,   LMUL,   MLEN)

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x)
{
	BLASLONG i=0, j=0;
	BLASLONG ix=0;
	FLOAT asumf=0.0;
	if (n <= 0 || inc_x <= 0) return(asumf);
        unsigned int gvl = 0;
        FLOAT_V_T v0, v1, v_zero,v_sum;
        FLOAT_V_T_M1 v_res;
        v_res = VFMVVF_FLOAT_M1(0, 1);

        if(inc_x == 1){
                BLASLONG n2 = n * 2;
                gvl = VSETVL(n2);
                v_zero = VFMVVF_FLOAT(0, gvl);
                if(gvl <= n2/2){
                        v_sum = VFMVVF_FLOAT(0, gvl);
                        for(i=0,j=0; i<n2/(gvl*2); i++){
                                v0 = VLEV_FLOAT(&x[j], gvl);
                                v_sum = VFADDVV_FLOAT(v_sum, v0, gvl);

                                v1 = VLEV_FLOAT(&x[j+gvl], gvl);
                                v_sum = VFADDVV_FLOAT(v_sum, v1, gvl);
                                j += gvl * 2;
                        }
                        v_res = VFREDSUMVS_FLOAT(v_sum, v_res, gvl);
                }
                for(;j<n2;){
                        gvl = VSETVL(n2-j);
                        v0 = VLEV_FLOAT(&x[j], gvl);
                        v_res = VFREDSUMVS_FLOAT(v0, v_res, gvl);
                        j += gvl;
                }
        }else{
                gvl = VSETVL(n);
                unsigned int stride_x = inc_x * sizeof(FLOAT) * 2;
                v_zero = VFMVVF_FLOAT(0, gvl);

                BLASLONG inc_xv = inc_x * 2 * gvl;
                v_sum = VFMVVF_FLOAT(0, gvl);
                for(i=0,j=0; i<n/gvl; i++){
                        v0 = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                        v_sum = VFADDVV_FLOAT(v_sum, v0, gvl);

                        v1 = VLSEV_FLOAT(&x[ix+1], stride_x, gvl);
                        v_sum = VFADDVV_FLOAT(v_sum, v1, gvl);

                        j += gvl;
                        ix += inc_xv;
                }
                v_res = VFREDSUMVS_FLOAT(v_sum, v_res, gvl);
                if(j<n){
                        gvl = VSETVL(n-j);
                        v0 = VLSEV_FLOAT(&x[ix], stride_x, gvl);

                        v1 = VLSEV_FLOAT(&x[ix+1], stride_x, gvl);
                        v_sum = VFADDVV_FLOAT(v0, v1, gvl);
                        v_res = VFREDSUMVS_FLOAT(v_sum, v_res, gvl);
                }
        }
        asumf = EXTRACT_FLOAT(v_res);
	return(asumf);
}


