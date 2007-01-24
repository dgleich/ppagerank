/*
 * David Gleich
 * 15 December 2006
 * Copyright, Stanford University
 */
 
/**
 * @file petsc_util.h
 * Prototypes for a set of PETSc utility routines.
 */
 
#ifndef PPAGERANK_PETSC_UTIL_H
#define PPAGERANK_PETSC_UTIL_H
 
#include "petsc.h"
#include "petscmat.h"
#include "petscvec.h"

PetscErrorCode MatLoadBSMAT(MPI_Comm comm,const char* filename, Mat *newmat);
PetscErrorCode MatLoadBVGraph(MPI_Comm comm,const char* filename, Mat *newmat);

PetscErrorCode MatGetNonzeroCount(Mat A, long long int *nzc, PetscInt *lnzc);
PetscErrorCode MatIsSquare(Mat A, PetscTruth *square);

PetscErrorCode MatBuildNonzeroRowIndicator(Mat A, Vec *ind);
PetscErrorCode MatBuildNonzeroColumnIndicator(Mat A, Vec *ind);

PetscErrorCode VecCreateForMatMult(Mat A, Vec *v);
PetscErrorCode VecCreateForMatMultTranspose(Mat A, Vec *v);
PetscErrorCode VecCreateForPossiblyTransposedMatrix(Mat A, Vec *v, PetscTruth trans);
PetscErrorCode VecNonzeroInv(Vec v);
PetscErrorCode VecNonzeroIndicator(Vec v);

PetscErrorCode VecCompatibleWithMatMult(Mat A, Vec v, PetscTruth *flg);
PetscErrorCode VecCompatibleWithMatMultTranspose(Mat A, Vec v, PetscTruth *flg);



PetscErrorCode PetscSynchronizedFEof(MPI_Comm comm,FILE *f,int *eof);
//PetscErrorCode VecNormDiff(Vec x, Vec y, NormType ntype, PetscScalar *norm

#endif /* PPAGERANK_PETSC_UTIL */

