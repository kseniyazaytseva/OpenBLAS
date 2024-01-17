/***************************************************************************
Copyright (C), 2023, KNS Group LLC (YADRO).

All Rights Reserved.


This software contains the intellectual property of YADRO or is licensed to YADRO

from third parties. Use of this software and the intellectual property contained

therein is expressly limited to the terms and conditions of the License Agreement

under which it is provided by YADRO.

*****************************************************************************/

#include "common.h"

double CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
{
        BLASLONG i=0, j=0;
        double dot = 0.0 ;

        if ( n < 1 )  return(dot);
        vfloat64m4_t vr;
        vfloat32m2_t vx, vy;
        unsigned int gvl = 0;
        vfloat64m1_t v_res, v_z0;
        gvl = vsetvlmax_e64m1();
        v_res = vfmv_v_f_f64m1(0, gvl);
        v_z0 = vfmv_v_f_f64m1(0, gvl);

        if(inc_x == 1 && inc_y == 1){
                gvl = vsetvl_e64m4(n);
                vr = vfmv_v_f_f64m4(0, gvl);
                for(i=0,j=0; i<n/gvl; i++){
                        vx = vle_v_f32m2(&x[j], gvl);
                        vy = vle_v_f32m2(&y[j], gvl);
                        vr = vfwmacc_vv_f64m4(vr, vx, vy, gvl);
                        j += gvl;
                }
                if(j > 0){
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);
                }
                //tail
                if(j < n){
                        gvl = vsetvl_e64m4(n-j);
                        vx = vle_v_f32m2(&x[j], gvl);
                        vy = vle_v_f32m2(&y[j], gvl);
                        vfloat64m4_t vz = vfmv_v_f_f64m4(0, gvl);
                        //vr = vfdot_vv_f32m2(vx, vy, gvl);
                        vr = vfwmacc_vv_f64m4(vz, vx, vy, gvl);
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);
                }
        }else if(inc_y == 1){
                gvl = vsetvl_e64m4(n);
                vr = vfmv_v_f_f64m4(0, gvl);
                 int stride_x = inc_x * sizeof(FLOAT);
                for(i=0,j=0; i<n/gvl; i++){
                        vx = vlse_v_f32m2(&x[j*inc_x], stride_x, gvl);
                        vy = vle_v_f32m2(&y[j], gvl);
                        vr = vfwmacc_vv_f64m4(vr, vx, vy, gvl);
                        j += gvl;
                }
                if(j > 0){
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
                //tail
                if(j < n){
                        gvl = vsetvl_e64m4(n-j);
                        vx = vlse_v_f32m2(&x[j*inc_x], stride_x, gvl);
                        vy = vle_v_f32m2(&y[j], gvl);
                        vfloat64m4_t vz = vfmv_v_f_f64m4(0, gvl);
                        //vr = vfdot_vv_f32m2(vx, vy, gvl);
                        vr = vfwmacc_vv_f64m4(vz, vx, vy, gvl);
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
        }else if(inc_x == 1){
                gvl = vsetvl_e64m4(n);
                vr = vfmv_v_f_f64m4(0, gvl);
                 int stride_y = inc_y * sizeof(FLOAT);
                for(i=0,j=0; i<n/gvl; i++){
                        vx = vle_v_f32m2(&x[j], gvl);
                        vy = vlse_v_f32m2(&y[j*inc_y], stride_y, gvl);
                        vr = vfwmacc_vv_f64m4(vr, vx, vy, gvl);
                        j += gvl;
                }
                if(j > 0){
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
                //tail
                if(j < n){
                        gvl = vsetvl_e64m4(n-j);
                        vx = vle_v_f32m2(&x[j], gvl);
                        vy = vlse_v_f32m2(&y[j*inc_y], stride_y, gvl);
                        vfloat64m4_t vz = vfmv_v_f_f64m4(0, gvl);
                        //vr = vfdot_vv_f32m2(vx, vy, gvl);
                        vr = vfwmacc_vv_f64m4(vz, vx, vy, gvl);
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
        }else{
                gvl = vsetvl_e64m4(n);
                vr = vfmv_v_f_f64m4(0, gvl);
                 int stride_x = inc_x * sizeof(FLOAT);
                 int stride_y = inc_y * sizeof(FLOAT);
                for(i=0,j=0; i<n/gvl; i++){
                        vx = vlse_v_f32m2(&x[j*inc_x], stride_x, gvl);
                        vy = vlse_v_f32m2(&y[j*inc_y], stride_y, gvl);
                        vr = vfwmacc_vv_f64m4(vr, vx, vy, gvl);
                        j += gvl;
                }
                if(j > 0){
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
                //tail
                if(j < n){
                        gvl = vsetvl_e64m4(n-j);
                        vx = vlse_v_f32m2(&x[j*inc_x], stride_x, gvl);
                        vy = vlse_v_f32m2(&y[j*inc_y], stride_y, gvl);
                        vfloat64m4_t vz = vfmv_v_f_f64m4(0, gvl);
                        //vr = vfdot_vv_f32m2(vx, vy, gvl);
                        vr = vfwmacc_vv_f64m4(vz, vx, vy, gvl);
                        v_res = vfredusum_vs_f64m4_f64m1(v_res, vr, v_z0, gvl);
                        dot += (double)vfmv_f_s_f64m1_f64(v_res);

                }
        }
        return(dot);
}
