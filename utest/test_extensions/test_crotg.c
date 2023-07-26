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

#ifdef BUILD_COMPLEX

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, zero_a)
{
    float sa[2] = {0.0f, 0.0f};
    float sb[2] = {1.0f, 1.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.0f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, zero_b)
{
    float sa[2] = {1.0f, 1.0f};
    float sb[2] = {0.0f, 0.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(1.0f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, zero_real)
{
    float sa[2] = {0.0f, 1.0f};
    float sb[2] = {0.0f, 1.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.7071f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.7071f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.41421f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, positive_real_positive_img)
{
    float sa[2] = {3.0f, 4.0f};
    float sb[2] = {4.0f, 6.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, negative_real_positive_img)
{
    float sa[2] = {-3.0f, 4.0f};
    float sb[2] = {-4.0f, 6.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, positive_real_negative_img)
{
    float sa[2] = {3.0f, -4.0f};
    float sb[2] = {4.0f, -6.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, negative_real_negative_img)
{
    float sa[2] = {-3.0f, -4.0f};
    float sb[2] = {-4.0f, -6.0f};
    float ss[2];
    float sc;
    BLASFUNC(crotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_zero_a)
{
    float sa[2] = {0.0f, 0.0f};
    float sb[2] = {1.0f, 1.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.0f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_zero_b)
{
    float sa[2] = {1.0f, 1.0f};
    float sb[2] = {0.0f, 0.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(1.0f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_zero_real)
{
    float sa[2] = {0.0f, 1.0f};
    float sb[2] = {0.0f, 1.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.7071f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.7071f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.41421f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_positive_real_positive_img)
{
    float sa[2] = {3.0f, 4.0f};
    float sb[2] = {4.0f, 6.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_negative_real_positive_img)
{
    float sa[2] = {-3.0f, 4.0f};
    float sb[2] = {-4.0f, 6.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_positive_real_negative_img)
{
    float sa[2] = {3.0f, -4.0f};
    float sb[2] = {4.0f, -6.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997f, sa[1], SINGLE_EPS);
}

/**
* Test crotg by comparing it against pre-calculated values
*/
CTEST(crotg, c_api_negative_real_negative_img)
{
    float sa[2] = {-3.0f, -4.0f};
    float sb[2] = {-4.0f, -6.0f};
    float ss[2];
    float sc;
    cblas_crotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.5698f, sc, SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82052f, ss[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558f, ss[1], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26498f, sa[0], SINGLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997f, sa[1], SINGLE_EPS);
}
#endif