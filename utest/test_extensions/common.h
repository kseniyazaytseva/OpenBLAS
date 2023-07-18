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
#endif