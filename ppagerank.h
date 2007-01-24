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

struct PageRankContext 
{
    PetscScalar tol;
    int maxiter;
    
    Mat P;
    PetscTruth trans;
    
    PetscTruth require_d;
    PetscTruth default_v;
    Vec v;
    Vec d;
};


PetscErrorCode MatNormalizeForPageRank(Mat A,PetscTruth trans,Vec *d);
PetscErrorCode ComputePageRank(Mat P, PetscTruth trans);
PetscErrorCode ComputePageRank_AlgPower(PageRankContext prc);
PetscErrorCode ComputePageRank_AlgLinsys(PageRankContext prc);
PetscErrorCode ComputePageRank_AlgArnoldi(PageRankContext prc);

#endif /* PPAGERANK_H */
