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
#include "common.h"


#define DATASIZE 100
#define INCREMENT 2

struct DATA_ZROT{
    double x_test[DATASIZE * INCREMENT * 2];
    double y_test[DATASIZE * INCREMENT * 2];
    double x_verify[DATASIZE * INCREMENT * 2];
    double y_verify[DATASIZE * INCREMENT * 2];
};

#ifdef BUILD_COMPLEX16
static struct DATA_ZROT data_zrot;


static void rand_generate(double *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / (double)RAND_MAX * 5.0;
}

/**
 * Comapare results computed by zdrot and zaxpby 
 * 
 * param n specifies size of vector x
 * param inc_x specifies increment of vector x
 * param inc_y specifies increment of vector y
 * param c specifies cosine
 * param s specifies sine
 * return norm of differences 
*/
static double check_zdrot(blasint n, blasint inc_x, blasint inc_y, double *c, double *s)
{
    blasint i;
    double norm = 0;
    double s_neg[] = {-s[0], s[1]};

    blasint inc_x_abs = labs(inc_x);
    blasint inc_y_abs = labs(inc_y);

    // Fill vectors x, y
    rand_generate(data_zrot.x_test, n * inc_x_abs * 2);
    rand_generate(data_zrot.y_test, n * inc_y_abs * 2);

    if (inc_x == 0 && inc_y == 0) {
        rand_generate(data_zrot.x_test, n * 2);
        rand_generate(data_zrot.y_test, n * 2);
    }

    // Copy vector x for zaxpby
    for (i = 0; i < n * inc_x_abs * 2; i++)
        data_zrot.x_verify[i] = data_zrot.x_test[i];

    // Copy vector y for zaxpby
    for (i = 0; i < n * inc_y_abs * 2; i++)
        data_zrot.y_verify[i] = data_zrot.y_test[i];
    
    // Find cx = c*x + s*y
    BLASFUNC(zaxpby)(&n, s, data_zrot.y_test, &inc_y, c, data_zrot.x_verify, &inc_x);

    // Find cy = -conjg(s)*x + c*y
    BLASFUNC(zaxpby)(&n, s_neg, data_zrot.x_test, &inc_x, c, data_zrot.y_verify, &inc_y);

    BLASFUNC(zdrot)(&n, data_zrot.x_test, &inc_x, data_zrot.y_test, &inc_y, c, s);

    // Find the differences between vector x caculated by zaxpby and zdrot
    for (i = 0; i < n * 2 * inc_x_abs; i++)
        data_zrot.x_test[i] -= data_zrot.x_verify[i];

    // Find the differences between vector y caculated by zaxpby and zdrot
    for (i = 0; i < n * 2 * inc_y_abs; i++)
        data_zrot.y_test[i] -= data_zrot.y_verify[i];

    // Find the norm of differences
    norm += BLASFUNC(dznrm2)(&n, data_zrot.x_test, &inc_x_abs);
    norm += BLASFUNC(dznrm2)(&n, data_zrot.y_test, &inc_y_abs);
    return (norm / 2);
}

/**
 * C API specific function
 * 
 * Comapare results computed by zdrot and zaxpby 
 * 
 * param n specifies size of vector x
 * param inc_x specifies increment of vector x
 * param inc_y specifies increment of vector y
 * param c specifies cosine
 * param s specifies sine
 * return norm of differences 
*/
static double c_api_check_zdrot(blasint n, blasint inc_x, blasint inc_y, double *c, double *s)
{
    blasint i;
    double norm = 0;
    double s_neg[] = {-s[0], s[1]};

    blasint inc_x_abs = labs(inc_x);
    blasint inc_y_abs = labs(inc_y);

    // Fill vectors x, y
    rand_generate(data_zrot.x_test, n * inc_x_abs * 2);
    rand_generate(data_zrot.y_test, n * inc_y_abs * 2);

    if (inc_x == 0 && inc_y == 0) {
        rand_generate(data_zrot.x_test, n * 2);
        rand_generate(data_zrot.y_test, n * 2);
    }

    // Copy vector x for zaxpby
    for (i = 0; i < n * inc_x_abs * 2; i++)
        data_zrot.x_verify[i] = data_zrot.x_test[i];

    // Copy vector y for zaxpby
    for (i = 0; i < n * inc_y_abs * 2; i++)
        data_zrot.y_verify[i] = data_zrot.y_test[i];
    
    // Find cx = c*x + s*y
    cblas_zaxpby(n, s, data_zrot.y_test, inc_y, c, data_zrot.x_verify, inc_x);

    // Find cy = -conjg(s)*x + c*y
    cblas_zaxpby(n, s_neg, data_zrot.x_test, inc_x, c, data_zrot.y_verify, inc_y);

    cblas_zdrot(n, data_zrot.x_test, inc_x, data_zrot.y_test, inc_y, c[0], s[0]);

    // Find the differences between vector x caculated by zaxpby and zdrot
    for (i = 0; i < n * 2 * inc_x_abs; i++)
        data_zrot.x_test[i] -= data_zrot.x_verify[i];

    // Find the differences between vector y caculated by zaxpby and zdrot
    for (i = 0; i < n * 2 * inc_y_abs; i++)
        data_zrot.y_test[i] -= data_zrot.y_verify[i];

    // Find the norm of differences
    norm += cblas_dznrm2(n, data_zrot.x_test, inc_x_abs);
    norm += cblas_dznrm2(n, data_zrot.y_test, inc_y_abs);
    return (norm / 2);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 0
 * Stride of vector y is 0
 * c = 1.0
 * s = 2.0
*/
CTEST(zrot, inc_x_0_inc_y_0)
{
    blasint n = 100;
    
    blasint inc_x = 0;
    blasint inc_y = 0;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, inc_x_1_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is -1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, inc_x_neg_1_inc_y_neg_1)
{
    blasint n = 100;
    
    blasint inc_x = -1;
    blasint inc_y = -1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 * c = 3.0
 * s = 2.0
*/
CTEST(zrot, inc_x_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {3.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, inc_x_neg_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, inc_x_1_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is -2
 * c = 2.0
 * s = 1.0
*/
CTEST(zrot, inc_x_1_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = -2;

    // Imaginary  part for zaxpby
    double c[] = {2.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 2.0
*/
CTEST(zrot, inc_x_2_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, inc_x_neg_2_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = -2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 0.0
 * s = 1.0
*/
CTEST(zrot, inc_x_2_inc_y_2_c_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {0.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 0.0
*/
CTEST(zrot, inc_x_2_inc_y_2_s_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {0.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 0
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, check_n_zero)
{
    blasint n = 0;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 * 
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 0
 * Stride of vector y is 0
 * c = 1.0
 * s = 2.0
*/
CTEST(zrot, c_api_inc_x_0_inc_y_0)
{
    blasint n = 100;
    
    blasint inc_x = 0;
    blasint inc_y = 0;

    // Imaginary  part for zaxpby
    double c[] = {3.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_1_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is -1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_neg_1_inc_y_neg_1)
{
    blasint n = 100;
    
    blasint inc_x = -1;
    blasint inc_y = -1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 * c = 3.0
 * s = 2.0
*/
CTEST(zrot, c_api_inc_x_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {3.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_neg_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_1_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is -2
 * c = 2.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_1_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = -2;

    // Imaginary  part for zaxpby
    double c[] = {2.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 2.0
*/
CTEST(zrot, c_api_inc_x_2_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {2.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_neg_2_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = -2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 0.0
 * s = 1.0
*/
CTEST(zrot, c_api_inc_x_2_inc_y_2_c_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {0.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0
 * s = 0.0
*/
CTEST(zrot, c_api_inc_x_2_inc_y_2_s_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {0.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}

/**
 * C API specific test
 *
 * Test zrot by comparing it with zaxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 0
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0
 * s = 1.0
*/
CTEST(zrot, c_api_check_n_zero)
{
    blasint n = 0;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for zaxpby
    double c[] = {1.0, 0.0};
    double s[] = {1.0, 0.0};

    double norm = c_api_check_zdrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0, norm, DOUBLE_EPS);
}
#endif