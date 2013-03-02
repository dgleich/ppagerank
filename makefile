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

CC=mpicxx

PETSC_DIR ?= /home/dgleich/dev/lib/petsc-2.3.2-p1
DGLEICH_DEV_DIR ?= /home/dgleich/dev
DGLEICH_LIB_DIR ?= /home/dgleich/dev/lib

CFLAGS    = -I.
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LDFLAGS   = -lz

# declare the all construct
all: ppagerank

# include the petsc makefile information
include ${PETSC_DIR}/bmake/common/base

PPAGERANK_LOCAL_OBJS = ppagerank.o ppagerank_main.o petsc_util.o bvgraph_matrix.o
PPAGERANK_REMOTE_OBJS = util/file.o util/string.o
PPAGERANK_REMOTE_OBJS_LINKFILES = file.o string.o
PPAGERANK_COMPILE_OBJS = $(PPAGERANK_LOCAL_OBJS) $(PPAGERANK_REMOTE_OBJS)
PPAGERANK_LINK_OBJS = $(PPAGERANK_LOCAL_OBJS) $(PPAGERANK_REMOTE_OBJS_LINKFILES) 

ppagerank: $(PPAGERANK_COMPILE_OBJS) chkopts
	mpicxx -o ppagerank $(PPAGERANK_LINK_OBJS) $(LDFLAGS) ${PETSC_LIB}
	${RM} $(PPAGERANK_LINK_OBJS)





