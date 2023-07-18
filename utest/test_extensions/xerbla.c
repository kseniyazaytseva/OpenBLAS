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

#include "common.h"

static int link_xerbla=TRUE;
static int lerr, _info, ok;
static char *rout;

static void F77_xerbla(char *srname, void *vinfo)
{
   int info=*(int*)vinfo;

   if (link_xerbla)
   {
      link_xerbla = 0;
      return;
   }

   if (rout != NULL && strcmp(rout, srname) != 0){
      printf("***** XERBLA WAS CALLED WITH SRNAME = <%s> INSTEAD OF <%s> *******\n", srname, rout);
      ok = FALSE;
   }

   if (info != _info){
      printf("***** XERBLA WAS CALLED WITH INFO = %d INSTEAD OF %d in %s *******\n",info, _info, srname);
      lerr = TRUE;
      ok = FALSE;
   } else lerr = FALSE;
}

/**  
* error function redefinition 
*/
int BLASFUNC(xerbla)(char *name, blasint *info, blasint length)
{
  F77_xerbla(name, info);
  return 0;
}

int check_error(void) {
   if (lerr == TRUE ) {
       printf("***** ILLEGAL VALUE OF PARAMETER NUMBER %d NOT DETECTED BY %s *****\n", _info, rout);
      ok = FALSE;
   }
   lerr = TRUE;
   return ok;
}

void set_xerbla(char* current_rout, int expected_info){
   if (link_xerbla) /* call these first to link */
      F77_xerbla(rout, &_info);

   ok = TRUE;
   lerr = TRUE;
   _info = expected_info;
   rout = current_rout;
}