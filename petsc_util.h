/*
 * David Gleich
 * 15 December 2006
 * Copyright, Stanford University
 */
 
/**
 * @file petsc_util.h
 * Prototypes for the 
 */
 
#include "petsc.h"
#include "petscmat.h"
#include "petscvec.h"

PetscErrorCode MatLoadBSMAT(MPI_Comm comm,const char* filename, Mat *newmat);
PetscErrorCode VecCreateForMat(Mat A, Vec *v);

