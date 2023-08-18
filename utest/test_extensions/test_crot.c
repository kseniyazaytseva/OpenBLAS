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

struct DATA_CROT{
    float x_test[DATASIZE * INCREMENT * 2];
    float y_test[DATASIZE * INCREMENT * 2];
    float x_verify[DATASIZE * INCREMENT * 2];
    float y_verify[DATASIZE * INCREMENT * 2];
};

#ifdef BUILD_COMPLEX
static struct DATA_CROT data_crot;


static void rand_generate(float *a, blasint n)
{
    blasint i;
    for (i = 0; i < n; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 5.0f;
}

/**
 * Comapare results computed by csrot and caxpby 
 * 
 * param n specifies size of vector x
 * param inc_x specifies increment of vector x
 * param inc_y specifies increment of vector y
 * param c specifies cosine
 * param s specifies sine
 * return norm of differences 
*/
static float check_csrot(blasint n, blasint inc_x, blasint inc_y, float *c, float *s)
{
    blasint i;
    float norm = 0;
    float s_neg[] = {-s[0], s[1]};

    blasint inc_x_abs = labs(inc_x);
    blasint inc_y_abs = labs(inc_y);

    // Fill vectors x, y
    rand_generate(data_crot.x_test, n * inc_x_abs * 2);
    rand_generate(data_crot.y_test, n * inc_y_abs * 2);

    if (inc_x == 0 && inc_y == 0) {
        rand_generate(data_crot.x_test, n * 2);
        rand_generate(data_crot.y_test, n * 2);
    }

    // Copy vector x for caxpby
    for (i = 0; i < n * inc_x_abs * 2; i++)
        data_crot.x_verify[i] = data_crot.x_test[i];

    // Copy vector y for caxpby
    for (i = 0; i < n * inc_y_abs * 2; i++)
        data_crot.y_verify[i] = data_crot.y_test[i];
    
    // Find cx = c*x + s*y
    BLASFUNC(caxpby)(&n, s, data_crot.y_test, &inc_y, c, data_crot.x_verify, &inc_x);

    // Find cy = -conjg(s)*x + c*y
    BLASFUNC(caxpby)(&n, s_neg, data_crot.x_test, &inc_x, c, data_crot.y_verify, &inc_y);

    BLASFUNC(csrot)(&n, data_crot.x_test, &inc_x, data_crot.y_test, &inc_y, c, s);

    // Find the differences between vector x caculated by caxpby and csrot
    for (i = 0; i < n * 2 * inc_x_abs; i++)
        data_crot.x_test[i] -= data_crot.x_verify[i];

    // Find the differences between vector y caculated by caxpby and csrot
    for (i = 0; i < n * 2 * inc_y_abs; i++)
        data_crot.y_test[i] -= data_crot.y_verify[i];

    // Find the norm of differences
    norm += BLASFUNC(scnrm2)(&n, data_crot.x_test, &inc_x_abs);
    norm += BLASFUNC(scnrm2)(&n, data_crot.y_test, &inc_y_abs);
    return (norm / 2);
}

/**
 * C API specific function
 * 
 * Comapare results computed by csrot and caxpby 
 * 
 * param n specifies size of vector x
 * param inc_x specifies increment of vector x
 * param inc_y specifies increment of vector y
 * param c specifies cosine
 * param s specifies sine
 * return norm of differences 
*/
static float c_api_check_csrot(blasint n, blasint inc_x, blasint inc_y, float *c, float *s)
{
    blasint i;
    float norm = 0;
    float s_neg[] = {-s[0], s[1]};

    blasint inc_x_abs = labs(inc_x);
    blasint inc_y_abs = labs(inc_y);

    // Fill vectors x, y
    rand_generate(data_crot.x_test, n * inc_x_abs * 2);
    rand_generate(data_crot.y_test, n * inc_y_abs * 2);

    if (inc_x == 0 && inc_y == 0) {
        rand_generate(data_crot.x_test, n * 2);
        rand_generate(data_crot.y_test, n * 2);
    }

    // Copy vector x for caxpby
    for (i = 0; i < n * inc_x_abs * 2; i++)
        data_crot.x_verify[i] = data_crot.x_test[i];

    // Copy vector y for caxpby
    for (i = 0; i < n * inc_y_abs * 2; i++)
        data_crot.y_verify[i] = data_crot.y_test[i];
    
    // Find cx = c*x + s*y
    cblas_caxpby(n, s, data_crot.y_test, inc_y, c, data_crot.x_verify, inc_x);

    // Find cy = -conjg(s)*x + c*y
    cblas_caxpby(n, s_neg, data_crot.x_test, inc_x, c, data_crot.y_verify, inc_y);

    cblas_csrot(n, data_crot.x_test, inc_x, data_crot.y_test, inc_y, c[0], s[0]);

    // Find the differences between vector x caculated by caxpby and csrot
    for (i = 0; i < n * 2 * inc_x_abs; i++)
        data_crot.x_test[i] -= data_crot.x_verify[i];

    // Find the differences between vector y caculated by caxpby and csrot
    for (i = 0; i < n * 2 * inc_y_abs; i++)
        data_crot.y_test[i] -= data_crot.y_verify[i];

    // Find the norm of differences
    norm += cblas_scnrm2(n, data_crot.x_test, inc_x_abs);
    norm += cblas_scnrm2(n, data_crot.y_test, inc_y_abs);
    return (norm / 2);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 0
 * Stride of vector y is 0
 * c = 1.0f
 * s = 2.0f
*/
CTEST(crot, inc_x_0_inc_y_0)
{
    blasint n = 100;
    
    blasint inc_x = 0;
    blasint inc_y = 0;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_1_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is -1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_neg_1_inc_y_neg_1)
{
    blasint n = 100;
    
    blasint inc_x = -1;
    blasint inc_y = -1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 * c = 3.0f
 * s = 2.0f
*/
CTEST(crot, inc_x_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {3.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_neg_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_1_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is -2
 * c = 2.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_1_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = -2;

    // Imaginary  part for caxpby
    float c[] = {2.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 2.0f
*/
CTEST(crot, inc_x_2_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_neg_2_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = -2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 0.0f
 * s = 1.0f
*/
CTEST(crot, inc_x_2_inc_y_2_c_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {0.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 0.0f
*/
CTEST(crot, inc_x_2_inc_y_2_s_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {0.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * FORTRAN API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 0
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, check_n_zero)
{
    blasint n = 0;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 * 
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 0
 * Stride of vector y is 0
 * c = 1.0f
 * s = 2.0f
*/
CTEST(crot, c_api_inc_x_0_inc_y_0)
{
    blasint n = 100;
    
    blasint inc_x = 0;
    blasint inc_y = 0;

    // Imaginary  part for caxpby
    float c[] = {3.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_1_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -1
 * Stride of vector y is -1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_neg_1_inc_y_neg_1)
{
    blasint n = 100;
    
    blasint inc_x = -1;
    blasint inc_y = -1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 1
 * c = 3.0f
 * s = 2.0f
*/
CTEST(crot, c_api_inc_x_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {3.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is -2
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_neg_2_inc_y_1)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is 2
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_1_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 1
 * Stride of vector y is -2
 * c = 2.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_1_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = 1;
    blasint inc_y = -2;

    // Imaginary  part for caxpby
    float c[] = {2.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 2.0f
*/
CTEST(crot, c_api_inc_x_2_inc_y_2)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {2.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_neg_2_inc_y_neg_2)
{
    blasint n = 100;
    
    blasint inc_x = -2;
    blasint inc_y = -2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 0.0f
 * s = 1.0f
*/
CTEST(crot, c_api_inc_x_2_inc_y_2_c_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {0.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 100
 * Stride of vector x is 2
 * Stride of vector y is 2
 * c = 1.0f
 * s = 0.0f
*/
CTEST(crot, c_api_inc_x_2_inc_y_2_s_zero)
{
    blasint n = 100;
    
    blasint inc_x = 2;
    blasint inc_y = 2;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {0.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}

/**
 * C API specific test
 *
 * Test crot by comparing it with caxpby.
 * Test with the following options:
 * 
 * Size of vectors x, y is 0
 * Stride of vector x is 1
 * Stride of vector y is 1
 * c = 1.0f
 * s = 1.0f
*/
CTEST(crot, c_api_check_n_zero)
{
    blasint n = 0;
    
    blasint inc_x = 1;
    blasint inc_y = 1;

    // Imaginary  part for caxpby
    float c[] = {1.0f, 0.0f};
    float s[] = {1.0f, 0.0f};

    float norm = c_api_check_csrot(n, inc_x, inc_y, c, s);
    ASSERT_DBL_NEAR_TOL(0.0f, norm, SINGLE_EPS);
}
#endif