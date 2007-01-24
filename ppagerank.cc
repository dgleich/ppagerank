/*
 * David Gleich
 * 24 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file ppagerank.cc
 * A set of PETSc functions to compute PageRank 
 */

#include "petsc_util.h"
#include "ppagerank_stages.h"

/**
 * Compute a PageRank vector for a PETSc Matrix P.
 * 
 * The ComputePageRank function loads all of its options from 
 * the command line.
 * 
 * @param P the matrix used for PageRank
 * @param trans true if the matrix is transposed
 * @return MPI_SUCCESS unless there was an error.
 */
#undef __FUNCT__
#define __FUNCT__ "ComputePageRank"
PetscErrorCode ComputePageRank(Mat P, PetscTruth trans)
{
    PetscErrorCode ierr;
    PetscTruth flag;
    MPI_Comm comm;
    ierr=PetscObjectGetComm((PetscObject)P,&comm); CHKERRQ(ierr);
    
    PetscInt M,N;
    ierr=MatGetSize(P,&M,&N);CHKERRQ(ierr);
    
    // make sure the matrix is square
    if (M!=N) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"The matrix P was not square.");
    }
    
    // grab the current alpha
    PetscScalar alpha=0.85;
    ierr=PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,&flag);CHKERRQ(ierr);
    
    PetscTruth noout=PETSC_FALSE;
    ierr=PetscOptionsHasName(PETSC_NULL,"-noout",&noout);CHKERRQ(ierr);
       
    PetscTruth default_v = PETSC_TRUE;
    char pvec_filename[PETSC_MAX_PATH_LEN];
    ierr=PetscOptionsGetString(PETSC_NULL,"-pvec",pvec_filename,PETSC_MAX_PATH_LEN,&flag);
        CHKERRQ(ierr);
    default_v = (PetscTruth)!flag;
    
    char algname[PETSC_MAX_PATH_LEN] = "power";
    ierr=PetscOptionsGetString(PETSC_NULL,"-alg",algname,PETSC_MAX_PATH_LEN,&flag);
        CHKERRQ(ierr);
        
    PetscTruth require_d = PETSC_FALSE;
    
    Vec v;
    if (!default_v) 
    {
        // they are not using the default vector, so we cannot optimize
        // for that case
        ierr=VecCreateForPossiblyTransposedMatrix(P,&v,trans);CHKERRQ(ierr);

        // load their vector
        
        // TODO check for sparse vector input and load that more efficiently
        PetscViewer viewer;
        ierr=PetscViewerBinaryOpen(comm,pvec_filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr=VecLoad(viewer,PETSC_NULL,&v);CHKERRQ(ierr);
    }
    
    // TODO Check v for a probability distribution
    
    // quick implementation of the power method    
    PetscInt maxiter = 1000;
    PetscScalar tol = 1e-7;
    
    PetscPrintf(comm,"Computing PageRank...\n");
    PetscPrintf(comm,"alg         = power\n");
    PetscPrintf(comm,"alpha       = %f\n", alpha);
    PetscPrintf(comm,"maxiter     = %i\n", maxiter);
    PetscPrintf(comm,"tol         = %g\n", tol);
    PetscScalar mat_norm_1,mat_norm_inf;
    ierr=MatNorm(P,NORM_1,&mat_norm_1);CHKERRQ(ierr);
    ierr=MatNorm(P,NORM_INFINITY,&mat_norm_inf);CHKERRQ(ierr);
    PetscPrintf(comm,"||P||_1     = %g\n",mat_norm_1);
    PetscPrintf(comm,"||P||_inf   = %g\n", mat_norm_inf);
    
    /*if (strcmp(alg,"power") == 0) {
        // run a quick implementation of the power method
        
    }*/
    

    // note that we don't need to worry about the orientation of the matrix 
    // because it is square by assumption so the column vectors are row 
    // vectors.
    Vec x, y;
    ierr=VecCreateForMatMult(P,&x);CHKERRQ(ierr);
    ierr=VecDuplicate(x,&y);CHKERRQ(ierr);
    if (default_v) {
        ierr=VecSet(x,1.0/(PetscScalar)N);CHKERRQ(ierr);
    } else {
        ierr=VecCopy(v,x);CHKERRQ(ierr);
    }
    
    PetscLogStagePush(STAGE_COMPUTE);
    
    for (PetscInt iter = 0; iter < maxiter; iter++) 
    {
        if (trans) {
            ierr=MatMult(P,x,y);CHKERRQ(ierr);
        } else {
            ierr=MatMultTranspose(P,x,y);CHKERRQ(ierr);
        }
        // compute y = c*y;
        ierr=VecScale(y,alpha);CHKERRQ(ierr);
        PetscScalar omega;
        ierr=VecNorm(y,NORM_1,&omega);
        omega = 1.0 - omega;
        if (default_v) {
            ierr=VecShift(y,omega/(PetscScalar)N);CHKERRQ(ierr);
        } else {
            ierr=VecAXPY(y,omega,v);CHKERRQ(ierr);
        }
           
        PetscScalar delta;
        // compute x = y - x
        PetscLogStagePush(STAGE_EVALUATE);
        ierr=VecAYPX(x,-1.0,y);CHKERRQ(ierr);
        ierr=VecNorm(x,NORM_1,&delta);CHKERRQ(ierr);
        PetscPrintf(comm,"%4i  %10.3e\n", iter+1, delta);
        PetscLogStagePop();
        
        if (delta < tol) {
            break;
        }
        ierr=VecCopy(y,x);CHKERRQ(ierr);
    }
    
    PetscLogStagePop();
    
    return (MPI_SUCCESS);
}

/**
 * Normalize the matrix A to be a row or column stochastic PageRank matrix.
 * If A is not tranposed, then the output A will be row stochastic.  
 * If A is transposed, than the output A will be column stochastic.
 * If the pointer to d is not null, then d will contain the dangling
 * node indicator vector.
 * 
 * The vector d should not be initialized.
 * 
 * This function must allocate (1) vector as an intermediate variable.
 * It will release this vector by the end.  The vector must be aligned
 * with the rows or columns of the matrix depending upon the trans
 * function.  
 * 
 * If the trans flag is set, the function must allocate (2) vectors
 * as intermediate variables.  
 * 
 */ 
#undef __FUNCT__
#define __FUNCT__ "MatNormalizeForPageRank" 
PetscErrorCode MatNormalizeForPageRank(Mat A,PetscTruth trans,Vec *d)
{
    PetscErrorCode ierr;
    // if they requested to get the dangling node indicator 
    if (d != PETSC_NULL) {
        ierr=VecCreateForPossiblyTransposedMatrix(A,d,trans);CHKERRQ(ierr);
    }
    
    if (trans) 
    {
        Vec row_align_vec;
        Vec col_align_vec;
        
        // we want a column aligned vector, not a row aligned vector
        ierr=VecCreateForMatMultTranspose(A,&col_align_vec);CHKERRQ(ierr);
        ierr=VecCreateForMatMult(A,&row_align_vec);CHKERRQ(ierr);
        
        // set the row aligned vector 
        ierr=VecSet(row_align_vec,1.0);CHKERRQ(ierr);
        
        // compute the column sums (M = A' => M'*e = A''*e = A*e = row sums!)  
        ierr=MatMultTranspose(A,row_align_vec,col_align_vec);CHKERRQ(ierr);
        
        // if they want the dangling nodes, we had better get them now
        if (d != PETSC_NULL) {
            ierr=VecDuplicate(col_align_vec,d);CHKERRQ(ierr);
            ierr=VecCopy(col_align_vec,*d);CHKERRQ(ierr);
            ierr=VecNonzeroIndicator(*d);CHKERRQ(ierr);
        }
        
        ierr=VecNonzeroInv(col_align_vec);CHKERRQ(ierr);
        ierr=MatDiagonalScale(A,PETSC_NULL,col_align_vec);CHKERRQ(ierr);
        
        ierr=VecDestroy(col_align_vec);CHKERRQ(ierr);
        ierr=VecDestroy(row_align_vec);CHKERRQ(ierr);

        // TODO fix PETSC_COMM_WORLD
        PetscPrintf(PETSC_COMM_WORLD,"\n");
        PetscPrintf(PETSC_COMM_WORLD,"** WARNING **\n");
        PetscPrintf(PETSC_COMM_WORLD,"Untested code path: %s, line %i.\n", __FILE__, __LINE__);
        PetscPrintf(PETSC_COMM_WORLD,"\n");
    }
    else
    {
        //
        // compute the row sums
        //
        Vec row_sums;        
        ierr=VecCreateForPossiblyTransposedMatrix(A,&row_sums,trans);CHKERRQ(ierr);
        
        // get the ownership range
        PetscInt Istart, Iend;
        ierr=MatGetOwnershipRange(A,&Istart, &Iend);CHKERRQ(ierr);
        
        // get the memory for the row sums vector and possibly d
        PetscScalar *d_data;
        if (d != PETSC_NULL) {
            ierr=VecGetArray(*d,&d_data);CHKERRQ(ierr);
        }
        PetscScalar *row_sums_data;
        ierr=VecGetArray(row_sums,&row_sums_data);CHKERRQ(ierr);
        
        // allocate a variable for the local vector index
        PetscInt local_i = 0;
        
        for (PetscInt i = Istart; i < Iend; ++i, ++local_i) 
        {
            PetscInt ncols=0;
            const PetscScalar *vals;
            ierr=MatGetRow(A,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
            PetscScalar d_entry = 0.0;
            PetscScalar sum = 0.0;
            for (PetscInt j = 0; j < ncols; j++) {
                // check to see if we should set the entry in d
                // TODO replace with machine epsilon
                if (d_entry == 0.0 && PetscAbsScalar(vals[j]) > 1e-16) {
                    d_entry = 1.0;
                }
                sum += vals[j];
            }
            ierr=MatRestoreRow(A,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
            
            if (d != PETSC_NULL) { d_data[local_i] = d_entry; }
            if (ncols > 0 && PetscAbsScalar(sum) > 1e-16) {
                row_sums_data[local_i] = 1.0/sum;
            } else {
                row_sums_data[local_i] = 0.0;
            }
            
            ierr=PetscLogFlops(ncols+1);
        }
        
        // restore the data backing all the vectors  
        ierr=VecRestoreArray(row_sums,&row_sums_data);CHKERRQ(ierr);
        if (d != PETSC_NULL) {
            ierr=VecRestoreArray(*d,&d_data);CHKERRQ(ierr);
        }
        
        // scale the matrix by the inverse outdegree of each element
        ierr=MatDiagonalScale(A,row_sums,PETSC_NULL);CHKERRQ(ierr);
        
        ierr=VecDestroy(row_sums);CHKERRQ(ierr);
    }
    
    return (MPI_SUCCESS);
}
