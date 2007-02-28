/*
 * David Gleich
 * 24 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file ppagerank.h
 * Prototypes for the ppagerank routines.
 */
 
#ifndef PPAGERANK_H
#define PPAGERANK_H
 
#include "petsc.h"
#include "petscmat.h"

EXTERN PetscErrorCode PRANKInitializePackage(const char[]);

#define PRANKPOWER "power"
#define PRANKARNOLDI "arnoldi"
#define PRANKLINSYS "linsys"

typedef struct _p_PR* PRANK;


// define a series of custom errors
#define PPAGERANK_ERR_ALG_UNKNOWN   PETSC_ERR_MAX_VALUE+1 /* the alg is not valid */

struct PageRankContext 
{
    PetscScalar tol;
    PetscInt maxiter;
    PetscInt N;
    
    Mat P;
    PetscTruth trans;
    
    PetscTruth require_d;
    PetscTruth default_v;
    Vec v;
    Vec d;
    
    PetscScalar alpha;

    PetscScalar inout_beta;
    PetscScalar inout_eta;

    PetscInt arnoldi_k;
    
    MPI_Comm comm;
};


PetscErrorCode MatNormalizeForPageRank(Mat A,PetscTruth trans,Vec *d);
PetscErrorCode ComputePageRank(Mat P, PetscTruth trans, Vec p);
PetscErrorCode ComputePageRank_AlgPower(PageRankContext prc,Vec p);
PetscErrorCode ComputePageRank_AlgLinsys(PageRankContext prc,Vec p);
PetscErrorCode ComputePageRank_AlgArnoldi(PageRankContext prc,Vec p);
PetscErrorCode ComputePageRank_AlgInOut(PageRankContext prc,Vec p);

extern int PR_COOKIE;
extern int PR_Solve;
extern int PR_Setup;

PetscErrorCode PageRankInitializePackage();


#endif /* PPAGERANK_H */
