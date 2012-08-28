/*
 * David Gleich
 * 29 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file pagacprvs.h
 * User interface for Graph Ranking Problems
 */
 
#if !defined(__PAGACPRVS_H)
#define __PAGACPRVS_H

#include "petsc.h"
#include "petscmat.h"

PETSC_EXTERN_CXX_BEGIN

extern PetscCookie PRVS_COOKIE;

typedef struct _p_PRVS* PRVS;

#define PRVSPOWER   "power"
#define PRVSARNOLDI "arnoldi"
#define PRVSLINSYS  "linsys"
#define PRVSType    const char*

typedef enum { PRVS_PAGERANK, PRVS_PAGERANK_DERIV } PRVSProblemType;

typedef enum { PRVS_DANGLING_V, PRVS_DANGLING_LOOP } PRVSDanglingFix;

PetscErrorCode PRVSCreate(MPI_Comm,PRVS*);
PetscErrorCode PRVSDestroy(PRVS);
PetscErrorCode PRVSView(PRVS,PetscViewer);

PetscErrorCode PRVSSetType(PRVS,PRVSType);
PetscErrorCode PRVSGetType(PRVS,PRVSType*);

PetscErrorCode PRVSSetProblemType(PRVS,PRVSProblemType);
PetscErrorCode PRVSGetProblemType(PRVS,PRVSProblemType*);

PetscErrorCode PRVSSetDanglingFix(PRVS,PRVSDanglingFix);
PetscErrorCode PRVSGetDanglingFix(PRVS,PRVSDanglingFix*);

PetscErrorCode PRVSSetFromOptions(PRVS);

PetscErrorCode PRVSSetTransitionMatrix(PRVS);
PetscErrorCode PRVSSetGraph(PRVS);

PetscErrorCode PRVSSetUp(PRVS);
PetscErrorCode PRVSSolve(PRVS);

PetscErrorCode PRVSSetTolerances(PRVS,PetscReal,PetscInt);
PetscErrorCode PRVSGetTolerances(PRVS,PetscReal*,PetscInt*);

typedef enum { /* converged */
               PRVS_CONVERGED_TOL       =  2,
               /* diverged */
               PRVS_DIVERGED_ITS        = -3,
               PRVS_DIVERGED_BREAKDOWN  = -4,
               /* working */
               PRVS_CONVERGED_ITERATING = 0} PRVSConvergedReason;
PetscErrorCode PRVSGetConvergedReason(PRVS,PRVSConvergedReason*);               

PetscErrorCode PRVSSetV(PRVS,Vec v);
PetscErrorCode PRVSGetV(PRVS,Vec *v);

PetscErrorCode PRVSSetInitialVector(PRVS, Vec x0);

#endif /* __PAGACPRVS_H */
