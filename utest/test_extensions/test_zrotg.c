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

#ifdef BUILD_COMPLEX16

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, zero_a)
{
    double sa[2] = {0.0, 0.0};
    double sb[2] = {1.0, 1.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.0, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific tests
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, zero_b)
{
    double sa[2] = {1.0, 1.0};
    double sb[2] = {0.0, 0.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(1.0, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, zero_real)
{
    double sa[2] = {0.0, 1.0};
    double sb[2] = {0.0, 1.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.70710678118654, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.70710678118654, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.41421356237309, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, positive_real_positive_img)
{
    double sa[2] = {3.0, 4.0};
    double sb[2] = {4.0, 6.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, negative_real_positive_img)
{
    double sa[2] = {-3.0, 4.0};
    double sb[2] = {-4.0, 6.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, positive_real_negative_img)
{
    double sa[2] = {3.0, -4.0};
    double sb[2] = {4.0, -6.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * Fortran API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, negative_real_negative_img)
{
    double sa[2] = {-3.0, -4.0};
    double sb[2] = {-4.0, -6.0};
    double ss[2];
    double sc;
    BLASFUNC(zrotg)(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_zero_a)
{
    double sa[2] = {0.0, 0.0};
    double sb[2] = {1.0, 1.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.0, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_zero_b)
{
    double sa[2] = {1.0, 1.0};
    double sb[2] = {0.0, 0.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(1.0, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.0, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_zero_real)
{
    double sa[2] = {0.0, 1.0};
    double sb[2] = {0.0, 1.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.70710678118654, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.70710678118654, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.0, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(1.41421356237309, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_positive_real_positive_img)
{
    double sa[2] = {3.0, 4.0};
    double sb[2] = {4.0, 6.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_negative_real_positive_img)
{
    double sa[2] = {-3.0, 4.0};
    double sb[2] = {-4.0, 6.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_positive_real_negative_img)
{
    double sa[2] = {3.0, -4.0};
    double sb[2] = {4.0, -6.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997150991369, sa[1], DOUBLE_EPS);
}

/**
 * C API specific test
 * Test zrotg by comparing it against pre-calculated values
 */
CTEST(zrotg, c_api_negative_real_negative_img)
{
    double sa[2] = {-3.0, -4.0};
    double sb[2] = {-4.0, -6.0};
    double ss[2];
    double sc;
    cblas_zrotg(sa, sb, &sc, ss);
    ASSERT_DBL_NEAR_TOL(0.56980288229818, sc, DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(0.82051615050939, ss[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-0.04558423058385, ss[1], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-5.26497863243527, sa[0], DOUBLE_EPS);
    ASSERT_DBL_NEAR_TOL(-7.01997150991369, sa[1], DOUBLE_EPS);
}
#endif