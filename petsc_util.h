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
PetscErrorCode MatLoadBVGraph(MPI_Comm comm,const char* filename, Mat *newmat);

PetscErrorCode MatGetNonzeroCount(Mat A, long long int *nzc, PetscInt *lnzc);
PetscErrorCode MatIsSquare(Mat A, PetscTruth *square);

PetscErrorCode VecCreateForMat(Mat A, Vec *v);
PetscErrorCode VecCreateForMatTranspose(Mat A, Vec *v);
PetscErrorCode VecNonzeroInv(Vec v);
PetscErrorCode VecNonzeroIndicator(Vec v);
//PetscErrorCode VecNormDiff(Vec x, Vec y, NormType ntype, PetscScalar *norm

