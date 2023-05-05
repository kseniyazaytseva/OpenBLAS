#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>
#ifdef complex
#undef complex
#endif
#ifdef I
#undef I
#endif

#if defined(_WIN64)
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

#ifdef LAPACK_ILP64
typedef BLASLONG blasint;
#if defined(_WIN64)
#define blasabs(x) llabs(x)
#else
#define blasabs(x) labs(x)
#endif
#else
typedef int blasint;
#define blasabs(x) abs(x)
#endif

typedef blasint integer;

typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
#ifdef _MSC_VER
static inline _Fcomplex Cf(complex *z) {_Fcomplex zz={z->r , z->i}; return zz;}
static inline _Dcomplex Cd(doublecomplex *z) {_Dcomplex zz={z->r , z->i};return zz;}
static inline _Fcomplex * _pCf(complex *z) {return (_Fcomplex*)z;}
static inline _Dcomplex * _pCd(doublecomplex *z) {return (_Dcomplex*)z;}
#else
static inline _Complex float Cf(complex *z) {return z->r + z->i*_Complex_I;}
static inline _Complex double Cd(doublecomplex *z) {return z->r + z->i*_Complex_I;}
static inline _Complex float * _pCf(complex *z) {return (_Complex float*)z;}
static inline _Complex double * _pCd(doublecomplex *z) {return (_Complex double*)z;}
#endif
#define pCf(z) (*_pCf(z))
#define pCd(z) (*_pCd(z))
typedef integer logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

typedef int flag;
typedef int ftnlen;
typedef int ftnint;

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (fabs(x))
#define f2cmin(a,b) ((a) <= (b) ? (a) : (b))
#define f2cmax(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (f2cmin(a,b))
#define dmax(a,b) (f2cmax(a,b))
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

#define abort_() { sig_die("Fortran abort routine called", 1); }
#define c_abs(z) (cabsf(Cf(z)))
#define c_cos(R,Z) { pCf(R)=ccos(Cf(Z)); }
#ifdef _MSC_VER
#define c_div(c, a, b) {Cf(c)._Val[0] = (Cf(a)._Val[0]/Cf(b)._Val[0]); Cf(c)._Val[1]=(Cf(a)._Val[1]/Cf(b)._Val[1]);}
#define z_div(c, a, b) {Cd(c)._Val[0] = (Cd(a)._Val[0]/Cd(b)._Val[0]); Cd(c)._Val[1]=(Cd(a)._Val[1]/df(b)._Val[1]);}
#else
#define c_div(c, a, b) {pCf(c) = Cf(a)/Cf(b);}
#define z_div(c, a, b) {pCd(c) = Cd(a)/Cd(b);}
#endif
#define c_exp(R, Z) {pCf(R) = cexpf(Cf(Z));}
#define c_log(R, Z) {pCf(R) = clogf(Cf(Z));}
#define c_sin(R, Z) {pCf(R) = csinf(Cf(Z));}
//#define c_sqrt(R, Z) {*(R) = csqrtf(Cf(Z));}
#define c_sqrt(R, Z) {pCf(R) = csqrtf(Cf(Z));}
#define d_abs(x) (fabs(*(x)))
#define d_acos(x) (acos(*(x)))
#define d_asin(x) (asin(*(x)))
#define d_atan(x) (atan(*(x)))
#define d_atn2(x, y) (atan2(*(x),*(y)))
#define d_cnjg(R, Z) { pCd(R) = conj(Cd(Z)); }
#define r_cnjg(R, Z) { pCf(R) = conjf(Cf(Z)); }
#define d_cos(x) (cos(*(x)))
#define d_cosh(x) (cosh(*(x)))
#define d_dim(__a, __b) ( *(__a) > *(__b) ? *(__a) - *(__b) : 0.0 )
#define d_exp(x) (exp(*(x)))
#define d_imag(z) (cimag(Cd(z)))
#define r_imag(z) (cimagf(Cf(z)))
#define d_int(__x) (*(__x)>0 ? floor(*(__x)) : -floor(- *(__x)))
#define r_int(__x) (*(__x)>0 ? floor(*(__x)) : -floor(- *(__x)))
#define d_lg10(x) ( 0.43429448190325182765 * log(*(x)) )
#define r_lg10(x) ( 0.43429448190325182765 * log(*(x)) )
#define d_log(x) (log(*(x)))
#define d_mod(x, y) (fmod(*(x), *(y)))
#define u_nint(__x) ((__x)>=0 ? floor((__x) + .5) : -floor(.5 - (__x)))
#define d_nint(x) u_nint(*(x))
#define u_sign(__a,__b) ((__b) >= 0 ? ((__a) >= 0 ? (__a) : -(__a)) : -((__a) >= 0 ? (__a) : -(__a)))
#define d_sign(a,b) u_sign(*(a),*(b))
#define r_sign(a,b) u_sign(*(a),*(b))
#define d_sin(x) (sin(*(x)))
#define d_sinh(x) (sinh(*(x)))
#define d_sqrt(x) (sqrt(*(x)))
#define d_tan(x) (tan(*(x)))
#define d_tanh(x) (tanh(*(x)))
#define i_abs(x) abs(*(x))
#define i_dnnt(x) ((integer)u_nint(*(x)))
#define i_len(s, n) (n)
#define i_nint(x) ((integer)u_nint(*(x)))
#define i_sign(a,b) ((integer)u_sign((integer)*(a),(integer)*(b)))
#define pow_dd(ap, bp) ( pow(*(ap), *(bp)))
#define pow_si(B,E) spow_ui(*(B),*(E))
#define pow_ri(B,E) spow_ui(*(B),*(E))
#define pow_di(B,E) dpow_ui(*(B),*(E))
#define pow_zi(p, a, b) {pCd(p) = zpow_ui(Cd(a), *(b));}
#define pow_ci(p, a, b) {pCf(p) = cpow_ui(Cf(a), *(b));}
#define pow_zz(R,A,B) {pCd(R) = cpow(Cd(A),*(B));}
#define s_cat(lpp, rpp, rnp, np, llp) { 	ftnlen i, nc, ll; char *f__rp, *lp; 	ll = (llp); lp = (lpp); 	for(i=0; i < (int)*(np); ++i) {         	nc = ll; 	        if((rnp)[i] < nc) nc = (rnp)[i]; 	        ll -= nc;         	f__rp = (rpp)[i]; 	        while(--nc >= 0) *lp++ = *(f__rp)++;         } 	while(--ll >= 0) *lp++ = ' '; }
#define s_cmp(a,b,c,d) ((integer)strncmp((a),(b),f2cmin((c),(d))))
#define s_copy(A,B,C,D) { int __i,__m; for (__i=0, __m=f2cmin((C),(D)); __i<__m && (B)[__i] != 0; ++__i) (A)[__i] = (B)[__i]; }
#define sig_die(s, kill) { exit(1); }
#define s_stop(s, n) {exit(0);}
static char junk[] = "\n@(#)LIBF77 VERSION 19990503\n";
#define z_abs(z) (cabs(Cd(z)))
#define z_exp(R, Z) {pCd(R) = cexp(Cd(Z));}
#define z_sqrt(R, Z) {pCd(R) = csqrt(Cd(Z));}
#define myexit_() break;
#define mycycle() continue;
#define myceiling(w) {ceil(w)}
#define myhuge(w) {HUGE_VAL}
//#define mymaxloc_(w,s,e,n) {if (sizeof(*(w)) == sizeof(double)) dmaxloc_((w),*(s),*(e),n); else dmaxloc_((w),*(s),*(e),n);}
#define mymaxloc(w,s,e,n) {dmaxloc_(w,*(s),*(e),n)}

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef logical (*L_fp)(...);
#else
typedef logical (*L_fp)();
#endif

static float spow_ui(float x, integer n) {
	float pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
static double dpow_ui(double x, integer n) {
	double pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#ifdef _MSC_VER
static _Fcomplex cpow_ui(complex x, integer n) {
	complex pow={1.0,0.0}; unsigned long int u;
		if(n != 0) {
		if(n < 0) n = -n, x.r = 1/x.r, x.i=1/x.i;
		for(u = n; ; ) {
			if(u & 01) pow.r *= x.r, pow.i *= x.i;
			if(u >>= 1) x.r *= x.r, x.i *= x.i;
			else break;
		}
	}
	_Fcomplex p={pow.r, pow.i};
	return p;
}
#else
static _Complex float cpow_ui(_Complex float x, integer n) {
	_Complex float pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#endif
#ifdef _MSC_VER
static _Dcomplex zpow_ui(_Dcomplex x, integer n) {
	_Dcomplex pow={1.0,0.0}; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x._Val[0] = 1/x._Val[0], x._Val[1] =1/x._Val[1];
		for(u = n; ; ) {
			if(u & 01) pow._Val[0] *= x._Val[0], pow._Val[1] *= x._Val[1];
			if(u >>= 1) x._Val[0] *= x._Val[0], x._Val[1] *= x._Val[1];
			else break;
		}
	}
	_Dcomplex p = {pow._Val[0], pow._Val[1]};
	return p;
}
#else
static _Complex double zpow_ui(_Complex double x, integer n) {
	_Complex double pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#endif
static integer pow_ii(integer x, integer n) {
	integer pow; unsigned long int u;
	if (n <= 0) {
		if (n == 0 || x == 1) pow = 1;
		else if (x != -1) pow = x == 0 ? 1/x : 0;
		else n = -n;
	}
	if ((n > 0) || !(n == 0 || x == 1 || x != -1)) {
		u = n;
		for(pow = 1; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
static integer dmaxloc_(double *w, integer s, integer e, integer *n)
{
	double m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
static integer smaxloc_(float *w, integer s, integer e, integer *n)
{
	float m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
static inline void cdotc_(complex *z, integer *n_, complex *x, integer *incx_, complex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Fcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conjf(Cf(&x[i]))._Val[0] * Cf(&y[i])._Val[0];
			zdotc._Val[1] += conjf(Cf(&x[i]))._Val[1] * Cf(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conjf(Cf(&x[i*incx]))._Val[0] * Cf(&y[i*incy])._Val[0];
			zdotc._Val[1] += conjf(Cf(&x[i*incx]))._Val[1] * Cf(&y[i*incy])._Val[1];
		}
	}
	pCf(z) = zdotc;
}
#else
	_Complex float zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conjf(Cf(&x[i])) * Cf(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conjf(Cf(&x[i*incx])) * Cf(&y[i*incy]);
		}
	}
	pCf(z) = zdotc;
}
#endif
static inline void zdotc_(doublecomplex *z, integer *n_, doublecomplex *x, integer *incx_, doublecomplex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Dcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conj(Cd(&x[i]))._Val[0] * Cd(&y[i])._Val[0];
			zdotc._Val[1] += conj(Cd(&x[i]))._Val[1] * Cd(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conj(Cd(&x[i*incx]))._Val[0] * Cd(&y[i*incy])._Val[0];
			zdotc._Val[1] += conj(Cd(&x[i*incx]))._Val[1] * Cd(&y[i*incy])._Val[1];
		}
	}
	pCd(z) = zdotc;
}
#else
	_Complex double zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conj(Cd(&x[i])) * Cd(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conj(Cd(&x[i*incx])) * Cd(&y[i*incy]);
		}
	}
	pCd(z) = zdotc;
}
#endif	
static inline void cdotu_(complex *z, integer *n_, complex *x, integer *incx_, complex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Fcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cf(&x[i])._Val[0] * Cf(&y[i])._Val[0];
			zdotc._Val[1] += Cf(&x[i])._Val[1] * Cf(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cf(&x[i*incx])._Val[0] * Cf(&y[i*incy])._Val[0];
			zdotc._Val[1] += Cf(&x[i*incx])._Val[1] * Cf(&y[i*incy])._Val[1];
		}
	}
	pCf(z) = zdotc;
}
#else
	_Complex float zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cf(&x[i]) * Cf(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cf(&x[i*incx]) * Cf(&y[i*incy]);
		}
	}
	pCf(z) = zdotc;
}
#endif
static inline void zdotu_(doublecomplex *z, integer *n_, doublecomplex *x, integer *incx_, doublecomplex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Dcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cd(&x[i])._Val[0] * Cd(&y[i])._Val[0];
			zdotc._Val[1] += Cd(&x[i])._Val[1] * Cd(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cd(&x[i*incx])._Val[0] * Cd(&y[i*incy])._Val[0];
			zdotc._Val[1] += Cd(&x[i*incx])._Val[1] * Cd(&y[i*incy])._Val[1];
		}
	}
	pCd(z) = zdotc;
}
#else
	_Complex double zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cd(&x[i]) * Cd(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cd(&x[i*incx]) * Cd(&y[i*incy]);
		}
	}
	pCd(z) = zdotc;
}
#endif
/*  -- translated by f2c (version 20000121).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/




/* Table of constant values */

static logical c_false = FALSE_;
static logical c_true = TRUE_;

/* > \brief \b CHSEIN */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CHSEIN + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/chsein.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/chsein.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/chsein.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CHSEIN( SIDE, EIGSRC, INITV, SELECT, N, H, LDH, W, VL, */
/*                          LDVL, VR, LDVR, MM, M, WORK, RWORK, IFAILL, */
/*                          IFAILR, INFO ) */

/*       CHARACTER          EIGSRC, INITV, SIDE */
/*       INTEGER            INFO, LDH, LDVL, LDVR, M, MM, N */
/*       LOGICAL            SELECT( * ) */
/*       INTEGER            IFAILL( * ), IFAILR( * ) */
/*       REAL               RWORK( * ) */
/*       COMPLEX            H( LDH, * ), VL( LDVL, * ), VR( LDVR, * ), */
/*      $                   W( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CHSEIN uses inverse iteration to find specified right and/or left */
/* > eigenvectors of a complex upper Hessenberg matrix H. */
/* > */
/* > The right eigenvector x and the left eigenvector y of the matrix H */
/* > corresponding to an eigenvalue w are defined by: */
/* > */
/* >              H * x = w * x,     y**h * H = w * y**h */
/* > */
/* > where y**h denotes the conjugate transpose of the vector y. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'R': compute right eigenvectors only; */
/* >          = 'L': compute left eigenvectors only; */
/* >          = 'B': compute both right and left eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] EIGSRC */
/* > \verbatim */
/* >          EIGSRC is CHARACTER*1 */
/* >          Specifies the source of eigenvalues supplied in W: */
/* >          = 'Q': the eigenvalues were found using CHSEQR; thus, if */
/* >                 H has zero subdiagonal elements, and so is */
/* >                 block-triangular, then the j-th eigenvalue can be */
/* >                 assumed to be an eigenvalue of the block containing */
/* >                 the j-th row/column.  This property allows CHSEIN to */
/* >                 perform inverse iteration on just one diagonal block. */
/* >          = 'N': no assumptions are made on the correspondence */
/* >                 between eigenvalues and diagonal blocks.  In this */
/* >                 case, CHSEIN must always perform inverse iteration */
/* >                 using the whole matrix H. */
/* > \endverbatim */
/* > */
/* > \param[in] INITV */
/* > \verbatim */
/* >          INITV is CHARACTER*1 */
/* >          = 'N': no initial vectors are supplied; */
/* >          = 'U': user-supplied initial vectors are stored in the arrays */
/* >                 VL and/or VR. */
/* > \endverbatim */
/* > */
/* > \param[in] SELECT */
/* > \verbatim */
/* >          SELECT is LOGICAL array, dimension (N) */
/* >          Specifies the eigenvectors to be computed. To select the */
/* >          eigenvector corresponding to the eigenvalue W(j), */
/* >          SELECT(j) must be set to .TRUE.. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix H.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] H */
/* > \verbatim */
/* >          H is COMPLEX array, dimension (LDH,N) */
/* >          The upper Hessenberg matrix H. */
/* >          If a NaN is detected in H, the routine will return with INFO=-6. */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >          The leading dimension of the array H.  LDH >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] W */
/* > \verbatim */
/* >          W is COMPLEX array, dimension (N) */
/* >          On entry, the eigenvalues of H. */
/* >          On exit, the real parts of W may have been altered since */
/* >          close eigenvalues are perturbed slightly in searching for */
/* >          independent eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VL */
/* > \verbatim */
/* >          VL is COMPLEX array, dimension (LDVL,MM) */
/* >          On entry, if INITV = 'U' and SIDE = 'L' or 'B', VL must */
/* >          contain starting vectors for the inverse iteration for the */
/* >          left eigenvectors; the starting vector for each eigenvector */
/* >          must be in the same column in which the eigenvector will be */
/* >          stored. */
/* >          On exit, if SIDE = 'L' or 'B', the left eigenvectors */
/* >          specified by SELECT will be stored consecutively in the */
/* >          columns of VL, in the same order as their eigenvalues. */
/* >          If SIDE = 'R', VL is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVL */
/* > \verbatim */
/* >          LDVL is INTEGER */
/* >          The leading dimension of the array VL. */
/* >          LDVL >= f2cmax(1,N) if SIDE = 'L' or 'B'; LDVL >= 1 otherwise. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VR */
/* > \verbatim */
/* >          VR is COMPLEX array, dimension (LDVR,MM) */
/* >          On entry, if INITV = 'U' and SIDE = 'R' or 'B', VR must */
/* >          contain starting vectors for the inverse iteration for the */
/* >          right eigenvectors; the starting vector for each eigenvector */
/* >          must be in the same column in which the eigenvector will be */
/* >          stored. */
/* >          On exit, if SIDE = 'R' or 'B', the right eigenvectors */
/* >          specified by SELECT will be stored consecutively in the */
/* >          columns of VR, in the same order as their eigenvalues. */
/* >          If SIDE = 'L', VR is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVR */
/* > \verbatim */
/* >          LDVR is INTEGER */
/* >          The leading dimension of the array VR. */
/* >          LDVR >= f2cmax(1,N) if SIDE = 'R' or 'B'; LDVR >= 1 otherwise. */
/* > \endverbatim */
/* > */
/* > \param[in] MM */
/* > \verbatim */
/* >          MM is INTEGER */
/* >          The number of columns in the arrays VL and/or VR. MM >= M. */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of columns in the arrays VL and/or VR required to */
/* >          store the eigenvectors (= the number of .TRUE. elements in */
/* >          SELECT). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array, dimension (N*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is REAL array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] IFAILL */
/* > \verbatim */
/* >          IFAILL is INTEGER array, dimension (MM) */
/* >          If SIDE = 'L' or 'B', IFAILL(i) = j > 0 if the left */
/* >          eigenvector in the i-th column of VL (corresponding to the */
/* >          eigenvalue w(j)) failed to converge; IFAILL(i) = 0 if the */
/* >          eigenvector converged satisfactorily. */
/* >          If SIDE = 'R', IFAILL is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] IFAILR */
/* > \verbatim */
/* >          IFAILR is INTEGER array, dimension (MM) */
/* >          If SIDE = 'R' or 'B', IFAILR(i) = j > 0 if the right */
/* >          eigenvector in the i-th column of VR (corresponding to the */
/* >          eigenvalue w(j)) failed to converge; IFAILR(i) = 0 if the */
/* >          eigenvector converged satisfactorily. */
/* >          If SIDE = 'L', IFAILR is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, i is the number of eigenvectors which */
/* >                failed to converge; see IFAILL and IFAILR for further */
/* >                details. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  Each eigenvector is normalized so that the element of largest */
/* >  magnitude has magnitude 1; here the magnitude of a complex number */
/* >  (x,y) is taken to be |x|+|y|. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
/* Subroutine */ int chsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, complex *h__, integer *ldh, complex *w, complex *
	vl, integer *ldvl, complex *vr, integer *ldvr, integer *mm, integer *
	m, complex *work, real *rwork, integer *ifaill, integer *ifailr, 
	integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1, 
	    i__2, i__3;
    real r__1, r__2;
    complex q__1, q__2;

    /* Local variables */
    real unfl;
    integer i__, k;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical leftv, bothv;
    real hnorm;
    integer kl;
    extern /* Subroutine */ int claein_(logical *, logical *, integer *, 
	    complex *, integer *, complex *, complex *, complex *, integer *, 
	    real *, real *, real *, integer *);
    integer kr, ks;
    complex wk;
    extern real slamch_(char *), clanhs_(char *, integer *, complex *,
	     integer *, real *);
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    extern logical sisnan_(real *);
    logical noinit;
    integer ldwork;
    logical rightv, fromqr;
    real smlnum;
    integer kln;
    real ulp, eps3;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Decode and test the input parameters. */

    /* Parameter adjustments */
    --select;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1 * 1;
    h__ -= h_offset;
    --w;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1 * 1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1 * 1;
    vr -= vr_offset;
    --work;
    --rwork;
    --ifaill;
    --ifailr;

    /* Function Body */
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;

    fromqr = lsame_(eigsrc, "Q");

    noinit = lsame_(initv, "N");

/*     Set M to the number of columns required to store the selected */
/*     eigenvectors. */

    *m = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (select[k]) {
	    ++(*m);
	}
/* L10: */
    }

    *info = 0;
    if (! rightv && ! leftv) {
	*info = -1;
    } else if (! fromqr && ! lsame_(eigsrc, "N")) {
	*info = -2;
    } else if (! noinit && ! lsame_(initv, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -5;
    } else if (*ldh < f2cmax(1,*n)) {
	*info = -7;
    } else if (*ldvl < 1 || leftv && *ldvl < *n) {
	*info = -10;
    } else if (*ldvr < 1 || rightv && *ldvr < *n) {
	*info = -12;
    } else if (*mm < *m) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHSEIN", &i__1, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     Set machine-dependent constants. */

    unfl = slamch_("Safe minimum");
    ulp = slamch_("Precision");
    smlnum = unfl * (*n / ulp);

    ldwork = *n;

    kl = 1;
    kln = 0;
    if (fromqr) {
	kr = 0;
    } else {
	kr = *n;
    }
    ks = 1;

    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (select[k]) {

/*           Compute eigenvector(s) corresponding to W(K). */

	    if (fromqr) {

/*              If affiliation of eigenvalues is known, check whether */
/*              the matrix splits. */

/*              Determine KL and KR such that 1 <= KL <= K <= KR <= N */
/*              and H(KL,KL-1) and H(KR+1,KR) are zero (or KL = 1 or */
/*              KR = N). */

/*              Then inverse iteration can be performed with the */
/*              submatrix H(KL:N,KL:N) for a left eigenvector, and with */
/*              the submatrix H(1:KR,1:KR) for a right eigenvector. */

		i__2 = kl + 1;
		for (i__ = k; i__ >= i__2; --i__) {
		    i__3 = i__ + (i__ - 1) * h_dim1;
		    if (h__[i__3].r == 0.f && h__[i__3].i == 0.f) {
			goto L30;
		    }
/* L20: */
		}
L30:
		kl = i__;
		if (k > kr) {
		    i__2 = *n - 1;
		    for (i__ = k; i__ <= i__2; ++i__) {
			i__3 = i__ + 1 + i__ * h_dim1;
			if (h__[i__3].r == 0.f && h__[i__3].i == 0.f) {
			    goto L50;
			}
/* L40: */
		    }
L50:
		    kr = i__;
		}
	    }

	    if (kl != kln) {
		kln = kl;

/*              Compute infinity-norm of submatrix H(KL:KR,KL:KR) if it */
/*              has not ben computed before. */

		i__2 = kr - kl + 1;
		hnorm = clanhs_("I", &i__2, &h__[kl + kl * h_dim1], ldh, &
			rwork[1]);
		if (sisnan_(&hnorm)) {
		    *info = -6;
		    return 0;
		} else if (hnorm > 0.f) {
		    eps3 = hnorm * ulp;
		} else {
		    eps3 = smlnum;
		}
	    }

/*           Perturb eigenvalue if it is close to any previous */
/*           selected eigenvalues affiliated to the submatrix */
/*           H(KL:KR,KL:KR). Close roots are modified by EPS3. */

	    i__2 = k;
	    wk.r = w[i__2].r, wk.i = w[i__2].i;
L60:
	    i__2 = kl;
	    for (i__ = k - 1; i__ >= i__2; --i__) {
		i__3 = i__;
		q__2.r = w[i__3].r - wk.r, q__2.i = w[i__3].i - wk.i;
		q__1.r = q__2.r, q__1.i = q__2.i;
		if (select[i__] && (r__1 = q__1.r, abs(r__1)) + (r__2 = 
			r_imag(&q__1), abs(r__2)) < eps3) {
		    q__1.r = wk.r + eps3, q__1.i = wk.i;
		    wk.r = q__1.r, wk.i = q__1.i;
		    goto L60;
		}
/* L70: */
	    }
	    i__2 = k;
	    w[i__2].r = wk.r, w[i__2].i = wk.i;

	    if (leftv) {

/*              Compute left eigenvector. */

		i__2 = *n - kl + 1;
		claein_(&c_false, &noinit, &i__2, &h__[kl + kl * h_dim1], ldh,
			 &wk, &vl[kl + ks * vl_dim1], &work[1], &ldwork, &
			rwork[1], &eps3, &smlnum, &iinfo);
		if (iinfo > 0) {
		    ++(*info);
		    ifaill[ks] = k;
		} else {
		    ifaill[ks] = 0;
		}
		i__2 = kl - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + ks * vl_dim1;
		    vl[i__3].r = 0.f, vl[i__3].i = 0.f;
/* L80: */
		}
	    }
	    if (rightv) {

/*              Compute right eigenvector. */

		claein_(&c_true, &noinit, &kr, &h__[h_offset], ldh, &wk, &vr[
			ks * vr_dim1 + 1], &work[1], &ldwork, &rwork[1], &
			eps3, &smlnum, &iinfo);
		if (iinfo > 0) {
		    ++(*info);
		    ifailr[ks] = k;
		} else {
		    ifailr[ks] = 0;
		}
		i__2 = *n;
		for (i__ = kr + 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + ks * vr_dim1;
		    vr[i__3].r = 0.f, vr[i__3].i = 0.f;
/* L90: */
		}
	    }
	    ++ks;
	}
/* L100: */
    }

    return 0;

/*     End of CHSEIN */

} /* chsein_ */

