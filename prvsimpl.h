/*
 * David Gleich
 * 29 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file pagacprvs.h
 * User interface for Graph Ranking Problems
 */
 
#if !defined(__PRVSIMPL_H)
#define __PRVSIMPL_H

#include "pagacprvs.h"

extern PetscFList PRVSList;
extern PetscEvent PRVS_Setup, PRVS_Solve;

typedef struct _PRVSOps *PRVSOps;

struct _PRVSOps {
    int (*solve)(PRVS);
    int (*setup)(PRVS);
    int (*setfromoptions)(PRVS);
    int (*publishoptions)(PRVS);
    int (*destroy)(PRVS);
    int (*view)(PRVS,PetscViewer);
};
    

struct _p_PRVS {
    PETSCHEADER(struct _PRVSOps);
    /* User Parameters */
    int maxit;                      /* maximum number of iterations */
    PetscReal tol;                  /* tolerance */
    PRVSProblemType problem_type;   /* which type of pagerank problem */
    
    /* PageRank problem parameters */
    PetscReal alpha;
    PetscTruth default_v;           
    Vec v;
    Mat P;
    
    /* Working data */
    Vec prvec;
    Vec prderiv;
    
    /* Solver dependent data */
    
    /* Status variables */
    PetscTruth prvec_available;     /* the PageRank vector is available */
    PetscTruth prderiv_available;     /* the PageRank derivative is available */
    
    PetscTruth setup_called;
    PRVSConvergedReason reason;

    void *data;                     /* placeholder for solver dependent data */
}
    

#endif /* __PRVSIMPL_H */
