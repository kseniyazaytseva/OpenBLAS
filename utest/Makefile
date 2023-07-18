UTEST_CHECK = 1
TOPDIR	= ..
DIR_EXT=test_extensions

override TARGET_ARCH=
override TARGET_MACH=

UTESTBIN=openblas_utest
UTESTEXTBIN=openblas_utest_ext

.PHONY : all
.NOTPARALLEL : all run_test $(UTESTBIN) $(UTESTEXTBIN)

include $(TOPDIR)/Makefile.system

OBJS=utest_main.o test_min.o test_amax.o test_ismin.o test_rotmg.o test_axpy.o test_dotu.o test_dsdot.o test_swap.o test_rot.o test_dnrm2.o
#test_rot.o test_swap.o test_axpy.o test_dotu.o test_dsdot.o test_fork.o
OBJS_EXT=utest_main.o $(DIR_EXT)/xerbla.o $(DIR_EXT)/test_isamin.o $(DIR_EXT)/test_idamin.o $(DIR_EXT)/test_icamin.o $(DIR_EXT)/test_izamin.o $(DIR_EXT)/test_ssum.o $(DIR_EXT)/test_dsum.o $(DIR_EXT)/test_scsum.o $(DIR_EXT)/test_dzsum.o $(DIR_EXT)/test_scamax.o $(DIR_EXT)/test_dzamax.o $(DIR_EXT)/test_samin.o $(DIR_EXT)/test_damin.o $(DIR_EXT)/test_scamin.o $(DIR_EXT)/test_dzamin.o

ifneq ($(NO_LAPACK), 1)
OBJS += test_potrs.o 
OBJS_EXT += $(DIR_EXT)/test_zspmv.o $(DIR_EXT)/test_cspmv.o
ifneq ($(NO_CBLAS), 1)
ifneq ($(NO_LAPACKE), 1)
OBJS += test_kernel_regress.o
endif
endif
endif

#this does not work with OpenMP nor with native Windows or Android threads
# FIXME TBD if this works on OSX, SunOS, POWER and zarch
ifeq ($(OSNAME), $(filter $(OSNAME),Linux CYGWIN_NT))
ifneq ($(USE_OPENMP), 1)
OBJS += test_fork.o
endif
OBJS += test_post_fork.o
endif

ifeq ($(C_COMPILER), PGI)
OBJS = utest_main2.o
endif
ifeq ($(C_COMPILER), SUN)
OBJS = utest_main2.o
endif
ifeq ($(OSNAME), AIX)
OBJS = utest_main2.o
endif

all : run_test

$(UTESTBIN): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ ../$(LIBNAME) $(EXTRALIB) $(FEXTRALIB)

$(UTESTEXTBIN): $(OBJS_EXT)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ ../$(LIBNAME) $(EXTRALIB) $(FEXTRALIB)

run_test: $(UTESTBIN) $(UTESTEXTBIN)
ifneq ($(CROSS), 1)
	./$(UTESTBIN)
	./$(UTESTEXTBIN)
endif

clean:
	-rm -f *.o $(UTESTBIN) $(UTESTEXTBIN)
	-rm -f $(DIR_EXT)/*.o

libs:
