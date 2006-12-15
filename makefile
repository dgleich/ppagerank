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

PETSC_DIR=/home/dgleich/dev/lib/petsc-2.3.2-p1
DGLEICH_DEV_DIR=/home/dgleich/dev/
DGLEICH_LIB_DIR=/home/dgleich/dev/lib

CFLAGS    = -I$(DGLEICH_DEV_DIR)/c++-util
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LDFLAGS   = -lz

# declare the all construct
all: ppagerank

# include the petsc makefile information
include ${PETSC_DIR}/bmake/common/base

ppagerank: ppagerank.o chkopts
	${CLINKER} -o ppagerank ppagerank.o  $(LDFLAGS) ${PETSC_LIB}
	${RM} ppagerank.o





