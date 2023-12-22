/***************************************************************************
Copyright (c) 2022, The OpenBLAS Project
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
#define VSETVL(n)               __riscv_vsetvl_e32m1(n)
#define FLOAT_V_T               vfloat32m1_t
#define FLOAT_VX2_T             vfloat32m1x2_t
#define FLOAT_VX4_T             vfloat32m1x4_t
#define FLOAT_VX8_T             vfloat32m1x8_t
#define VLEV_FLOAT              __riscv_vle32_v_f32m1
#define VSEV_FLOAT              __riscv_vse32_v_f32m1
#define VLSSEG2_FLOAT           __riscv_vlsseg2e32_v_f32m1x2
#define VSSEG2_FLOAT            __riscv_vsseg2e32_v_f32m1x2
#define VLSSEG4_FLOAT           __riscv_vlsseg4e32_v_f32m1x4
#define VSSEG4_FLOAT            __riscv_vsseg4e32_v_f32m1x4
#define VLSSEG8_FLOAT           __riscv_vlsseg8e32_v_f32m1x8
#define VSSEG8_FLOAT            __riscv_vsseg8e32_v_f32m1x8
#else
#define VSETVL(n)               __riscv_vsetvl_e64m1(n)
#define FLOAT_V_T               vfloat64m1_t
#define FLOAT_VX2_T             vfloat64m1x2_t
#define FLOAT_VX4_T             vfloat64m1x4_t
#define FLOAT_VX8_T             vfloat64m1x8_t
#define VLEV_FLOAT              __riscv_vle64_v_f64m1
#define VSEV_FLOAT              __riscv_vse64_v_f64m1
#define VLSSEG2_FLOAT           __riscv_vlsseg2e64_v_f64m1x2
#define VSSEG2_FLOAT            __riscv_vsseg2e64_v_f64m1x2
#define VLSSEG4_FLOAT           __riscv_vlsseg4e64_v_f64m1x4
#define VSSEG4_FLOAT            __riscv_vsseg4e64_v_f64m1x4
#define VLSSEG8_FLOAT           __riscv_vlsseg8e64_v_f64m1x8
#define VSSEG8_FLOAT            __riscv_vsseg8e64_v_f64m1x8
#endif

// Optimizes the implementation in ../generic/gemm_ncopy_8.c

int CNAME(BLASLONG m, BLASLONG n, FLOAT *a, BLASLONG lda, FLOAT *b)
{
    BLASLONG i, j;

    FLOAT *a_offset;
    FLOAT *a_offset1;
    FLOAT *b_offset;

    FLOAT_V_T v1;
    FLOAT_VX2_T vx2;
    FLOAT_VX4_T vx4;
    FLOAT_VX8_T vx8;

    size_t vl;

    //fprintf(stderr, "gemm_ncopy_8 m=%ld n=%ld lda=%ld\n", m, n, lda);

    a_offset = a;
    b_offset = b;

    for(j = (n >> 3); j > 0; j--) {
        a_offset1  = a_offset;
        a_offset += 8 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL(i);

            vx8 = VLSSEG8_FLOAT(a_offset1, lda * sizeof(FLOAT), vl);
            VSSEG8_FLOAT(b_offset, vx8, vl);

            a_offset1 += vl;
            b_offset += vl*8;
        }
    }

    if (n & 4) {
        a_offset1  = a_offset;
        a_offset += 4 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL(i);

            vx4 = VLSSEG4_FLOAT(a_offset1, lda * sizeof(FLOAT), vl);
            VSSEG4_FLOAT(b_offset, vx4, vl);

            a_offset1 += vl;
            b_offset += vl*4;
        }
    }

    if (n & 2) {
        a_offset1  = a_offset;
        a_offset += 2 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL(i);

            vx2 = VLSSEG2_FLOAT(a_offset1, lda * sizeof(FLOAT), vl);
            VSSEG2_FLOAT(b_offset, vx2, vl);

            a_offset1 += vl;
            b_offset += vl*2;
        }
    }

    if (n & 1) {
        a_offset1  = a_offset;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL(i);

            v1 = VLEV_FLOAT(a_offset1, vl);

            VSEV_FLOAT(b_offset, v1, vl);

            a_offset1 += vl;
            b_offset += vl;
        }
    }

    return 0;
}
