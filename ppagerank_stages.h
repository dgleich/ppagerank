/*
 * David Gleich
 * 24 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file ppagerank_stages.h
 * Prototypes for the 
 */

#ifndef PPAGERANK_STAGES_H
#define PPAGERANK_STAGES_H

#include "petsc.h"

#if defined (PPAGERANK_STAGES_DECLARE)
#define PPAGERANK_STAGES_EXTERN_TYPE
#else
#define PPAGERANK_STAGES_EXTERN_TYPE extern
#endif /* PPAGERANK_STAGES_DECLARE */

PPAGERANK_STAGES_EXTERN_TYPE int STAGE_LOAD;
PPAGERANK_STAGES_EXTERN_TYPE int STAGE_COMPUTE;
PPAGERANK_STAGES_EXTERN_TYPE int STAGE_EVALUATE;
PPAGERANK_STAGES_EXTERN_TYPE int STAGE_OUTPUT;

#if defined (PPAGERANK_STAGES_DECLARE)
void RegisterStages() {
    PetscLogStageRegister(&STAGE_LOAD,"Load Data");
    PetscLogStageRegister(&STAGE_COMPUTE,"Compute");
    PetscLogStageRegister(&STAGE_EVALUATE,"Evaluation");
    PetscLogStageRegister(&STAGE_OUTPUT,"Output Data");
}
#endif /* PPAGERANK_STAGES_DECLARE */


#endif /*PPAGERANK_STAGES_H*/
