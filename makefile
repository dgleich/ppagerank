#
# ppagerank makefile
#
# History
#   13 Dec 2006: Initial revision
#

#
# David Gleich
# 13 December 2006
# Copyright, Stanford University
# 

PETSC_DIR ?= /home/dgleich/dev/lib/petsc-2.3.2-p1
DGLEICH_DEV_DIR ?= /home/dgleich/dev/
DGLEICH_LIB_DIR ?= /home/dgleich/dev/lib

CFLAGS    = -I$(DGLEICH_DEV_DIR)/c++-util
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LDFLAGS   = -lz

# declare the all construct
all: ppagerank

# include the petsc makefile information
include ${PETSC_DIR}/bmake/common/base

PPAGERANK_OBJS = ppagerank.o petsc_util.o

ppagerank: $(PPAGERANK_OBJS) chkopts
	${CLINKER} -o ppagerank $(PPAGERANK_OBJS) $(LDFLAGS) ${PETSC_LIB}
	${RM} $(PPAGERANK_OBJS)





