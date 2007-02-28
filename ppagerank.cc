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
#include "ppagerank.h"

#include "petscblaslapack.h"

#include <petscsys.h>
#include <string.h>


PetscErrorCode PageRankInitializePackage()
{
    static PetscTruth initialized = PETSC_FALSE;
    //PetscErrorCode ierr;
    
    PetscFunctionBegin;
    if (initialized) { PetscFunctionReturn(0); }
    
    PetscFunctionReturn(0);
}

PetscErrorCode PageRankMult(PageRankContext prc, Vec x, Vec y);
PetscErrorCode PageRankDanglingMult(PageRankContext prc, Vec x, Vec y);

/**
 * Compute a PageRank vector for a PETSc Matrix P.
 * 
 * The ComputePageRank function loads all of its options from 
 * the command line.
 * 
 * @param P the matrix used for PageRank
 * @param trans true if the matrix is transposed
 * @param x the output vector x
 * @return MPI_SUCCESS unless there was an error.
 */
#undef __FUNCT__
#define __FUNCT__ "ComputePageRank"
PetscErrorCode ComputePageRank(Mat P, PetscTruth trans, Vec x)
{
    PetscErrorCode ierr;
    PetscTruth flag;
    MPI_Comm comm;
    
    PetscFunctionBegin;
    
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
        
    // quick implementation of the power method    
    PetscInt maxiter = 10000;
    ierr=PetscOptionsGetInt(PETSC_NULL,"-maxiter",&maxiter,&flag);CHKERRQ(ierr);
    
    PetscScalar tol=1e-7;
    ierr=PetscOptionsGetScalar(PETSC_NULL,"-tol",&tol,&flag);CHKERRQ(ierr);
    
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
    
    PetscPrintf(comm,"Computing PageRank...\n");
    PetscPrintf(comm,"alg         = %s\n", algname);
    PetscPrintf(comm,"alpha       = %f\n", alpha);
    PetscPrintf(comm,"maxiter     = %i\n", maxiter);
    PetscPrintf(comm,"tol         = %g\n", tol);
    PetscScalar mat_norm_1,mat_norm_inf;
    ierr=MatNorm(P,NORM_1,&mat_norm_1);CHKERRQ(ierr);
    ierr=MatNorm(P,NORM_INFINITY,&mat_norm_inf);CHKERRQ(ierr);
    PetscPrintf(comm,"||P||_1     = %g\n",mat_norm_1);
    PetscPrintf(comm,"||P||_inf   = %g\n", mat_norm_inf);

    PageRankContext prc = {0};
    prc.alpha = alpha;
    prc.tol = tol;
    prc.maxiter = maxiter;
    prc.P = P;
    prc.v = v;
    prc.default_v = default_v;
    prc.N = N;
    prc.trans = trans;
    prc.comm = comm;
    
    if (strcmp(algname,"arnoldi") == 0) {
        ierr = ComputePageRank_AlgArnoldi(prc,x); CHKERRQ(ierr);
    } else if (strcmp(algname,"power") == 0) {
        // run a quick implementation of the power method
        ierr = ComputePageRank_AlgPower(prc,x); CHKERRQ(ierr);
    } else if (strcmp(algname,"linsys") == 0) {
        //ierr = ComputePageRank_AlgLinsys(prc,x); CHKERRQ(ierr);
    } else if (strcmp(algname,"inout") == 0) {
        // run a quick implementation of the inner/outer iteration
        ierr=ComputePageRank_AlgInOut(prc,x); CHKERRQ(ierr);
    } else {
        SETERRQ1(PETSC_ERR_SUP,"Unknown algorithm: %s.", algname);
    }


    
    PetscFunctionReturn(0);
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

#undef __FUNCT__
#define __FUNCT__ "ComputePageRank_AlgPower"
PetscErrorCode ComputePageRank_AlgPower(PageRankContext prc,Vec p)
{
    PetscErrorCode ierr;
    
    // TODO add code to check the vector p 
    // to make sure it is a valid Pagerank vector.
    
    // note that we don't need to worry about the orientation of the matrix 
    // because it is square by assumption so the column vectors are row 
    // vectors.
    Vec x = p;
    Vec y;
    ierr=VecDuplicate(x,&y);CHKERRQ(ierr);
    if (prc.default_v) {
        ierr=VecSet(x,1.0/(PetscScalar)prc.N);CHKERRQ(ierr);
    } else {
        ierr=VecCopy(prc.v,x);CHKERRQ(ierr);
    }
    
    PetscLogStagePush(STAGE_COMPUTE);
    
    for (PetscInt iter = 0; iter < prc.maxiter; iter++) 
    {
        if (prc.trans) {
            ierr=MatMult(prc.P,x,y);CHKERRQ(ierr);
        } else {
            ierr=MatMultTranspose(prc.P,x,y);CHKERRQ(ierr);
        }
        // compute y = c*y;
        ierr=VecScale(y,prc.alpha);CHKERRQ(ierr);
        PetscScalar omega;
        ierr=VecNorm(y,NORM_1,&omega);
        omega = 1.0 - omega;
        if (prc.default_v) {
            ierr=VecShift(y,omega/(PetscScalar)prc.N);CHKERRQ(ierr);
        } else {
            ierr=VecAXPY(y,omega,prc.v);CHKERRQ(ierr);
        }
           
        PetscScalar delta;
        // compute x = y - x
        PetscLogStagePush(STAGE_EVALUATE);
        ierr=VecAYPX(x,-1.0,y);CHKERRQ(ierr);
        ierr=VecNorm(x,NORM_1,&delta);CHKERRQ(ierr);
        PetscPrintf(prc.comm,"%4i  %10.3e\n", iter+1, delta);
        PetscLogStagePop();
        
        if (delta < prc.tol) {
            break;
        }
        ierr=VecCopy(y,x);CHKERRQ(ierr);
    }
    
    ierr = VecDestroy(y);
    
    PetscLogStagePop();
    
    return (MPI_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePageRank_AlgInOut"
PetscErrorCode ComputePageRank_AlgInOut(PageRankContext prc,Vec p)
{
    PetscTruth flag;
    PetscErrorCode ierr;
  
    PetscScalar beta=prc.alpha/2;
    ierr=PetscOptionsGetScalar(PETSC_NULL,"-inout_beta",&beta,&flag);CHKERRQ(ierr);
  
    PetscScalar eta=1e-2;
    ierr=PetscOptionsGetScalar(PETSC_NULL,"-inout_eta",&eta,&flag);CHKERRQ(ierr);
  
    PetscInt max_inner_iter = 1000;
    ierr=PetscOptionsGetInt(PETSC_NULL,"-inout_max_inner_iter",&max_inner_iter,&flag);
        CHKERRQ(ierr);

    // TODO add code to check the vector p
    // to make sure it is a valid Pagerank vector.

    // note that we don't need to worry about the orientation of the matrix
    // because it is square by assumption so the column vectors are row
    // vectors.
    Vec x = p;
    Vec y;
    ierr=VecDuplicate(x,&y);CHKERRQ(ierr);
    if (prc.default_v) {
        ierr=VecSet(x,1.0/(PetscScalar)prc.N);CHKERRQ(ierr);
    } else {
        ierr=VecCopy(prc.v,x);CHKERRQ(ierr);
    }
    
    Vec inner_rhs;
    ierr=VecDuplicate(y,&inner_rhs);CHKERRQ(ierr);
    
    PetscTruth inner_iteration = PETSC_TRUE;
    PetscScalar delta;

    PetscLogStagePush(STAGE_COMPUTE);
    
    ierr=PageRankDanglingMult(prc,x,y);CHKERRQ(ierr);
    
        
    for (PetscInt iter = 0; iter < prc.maxiter; iter++)
    {
        if (!inner_iteration) {
            // just do a power iteration
            if (prc.trans) {
                ierr=MatMult(prc.P,x,y);CHKERRQ(ierr);
            } else {
                ierr=MatMultTranspose(prc.P,x,y);CHKERRQ(ierr);
            }
            // compute y = c*y;
            ierr=VecScale(y,prc.alpha);CHKERRQ(ierr);
            PetscScalar omega;
            ierr=VecNorm(y,NORM_1,&omega);
            omega = 1.0 - omega;
            if (prc.default_v) {
                ierr=VecShift(y,omega/(PetscScalar)prc.N);CHKERRQ(ierr);
            } else {
                ierr=VecAXPY(y,omega,prc.v);CHKERRQ(ierr);
            }
            
            PetscLogStagePush(STAGE_EVALUATE);
            ierr=VecAYPX(x,-1.0,y);CHKERRQ(ierr);
            ierr=VecNorm(x,NORM_1,&delta);CHKERRQ(ierr);
            PetscLogStagePop();
            
            ierr=VecCopy(y,x);CHKERRQ(ierr);
        }
        if (inner_iteration) {
            // form the rhs of the inner iteration ((a-b)Px + (1-a)v)
	       ierr=VecCopy(y,inner_rhs);  // inner_rhs <- y
	       ierr=VecScale(inner_rhs,prc.alpha-beta);CHKERRQ(ierr); // inner_rhs = (alpha - beta)*y
           
           // inner_rhs <- (alpha-beta)*y + (1-alpha)*v
	       if (prc.default_v) {
	           ierr=VecShift(inner_rhs,(1.0-prc.alpha)/(PetscScalar)prc.N);CHKERRQ(ierr);
	       } else {
	           ierr=VecAXPY(inner_rhs,(1.0-prc.alpha),prc.v);CHKERRQ(ierr);
	       }
        
            // begin the inner iteration
            for (PetscInt inner_iter=0; inner_iter < max_inner_iter && inner_iteration; inner_iter++) {
                // x <- beta*y + inner_rhs
                ierr=VecWAXPY(x,beta,y,inner_rhs);CHKERRQ(ierr);
    
                // y <- P'*x
                ierr=PageRankDanglingMult(prc,x,y);CHKERRQ(ierr);
                
                // compute delta = ||inner_rhs + beta*y - x||_1
                PetscLogStagePush(STAGE_EVALUATE);
                //ierr=VecAYPX(x,-1.0,y);CHKERRQ(ierr);
                ierr=VecAXPBY(x,beta,-1.0,y);CHKERRQ(ierr);  // x <- by - x
                ierr=VecAXPY(x,1.0,inner_rhs);CHKERRQ(ierr); // x <- x + f = (f + by - x)
                ierr=VecNorm(x,NORM_1,&delta);CHKERRQ(ierr); // delta = ||x||
                //PetscPrintf(prc.comm,"%4i  %10.3e\n", inner_iter+1, delta);
                PetscLogStagePop();
                if (delta < eta) {
                    if (inner_iter == 0) { 
                        inner_iteration = PETSC_FALSE; 
                        
                    }
                    break;
                }
            } 
            
            // note that x here is not set to the correct entry
            //  x = (f + by - x)
            // so undo the changes to x
            //
            // we changed x so we didn't have to use another vector
            // to compute the norm
            ierr=VecAXPY(x,-1.0,inner_rhs);CHKERRQ(ierr); // x <- x - f = (by - x)
            ierr=VecAXPBY(x,beta,-1.0,y);CHKERRQ(ierr);  // x <- -x + by = x
                      
            PetscLogStagePush(STAGE_EVALUATE);
            // reuse the f vector to compute the norm here (f = inner_rhs)
            ierr=VecWAXPY(inner_rhs,-(prc.alpha),y,x);CHKERRQ(ierr); 
            if (prc.default_v) {
                ierr=VecShift(inner_rhs,-(1-prc.alpha)/(PetscScalar)prc.N);CHKERRQ(ierr);
            } else {
                ierr=VecAXPY(inner_rhs,-(1-prc.alpha),prc.v);CHKERRQ(ierr);
            }
            ierr=VecNorm(inner_rhs,NORM_1,&delta);CHKERRQ(ierr);
            PetscLogStagePop();
            
            if (inner_iteration == PETSC_FALSE) {
                // if we jump out of the iteration, set x 
                // to the next iterate
                ierr=VecCopy(y,x);CHKERRQ(ierr);
                ierr=VecScale(x,prc.alpha);CHKERRQ(ierr);
                
                if (prc.default_v) {
                    ierr=VecShift(x,(1.0-prc.alpha)/(PetscScalar)prc.N);CHKERRQ(ierr);
                } else {
                    ierr=VecAXPY(x,(1.0-prc.alpha),prc.v);CHKERRQ(ierr);
                }
            }
            
        }
        
        PetscLogStagePush(STAGE_EVALUATE);
        PetscPrintf(prc.comm,"%4i  %10.3e %1i\n", iter+1, delta, inner_iteration);
        PetscLogStagePop();
        
    
        if (delta < prc.tol) {
    	   break;
        }
        
        
    }

  

  ierr = VecDestroy(y);

  PetscLogStagePop();

  return (MPI_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePageRank_AlgArnoldi"
PetscErrorCode ComputePageRank_AlgArnoldi(PageRankContext prc,Vec p)
{
    PetscErrorCode ierr;
    
    PetscInt k = 8;
    
    // I figured out an implementation that doesn't require the dangling vector
    // if (prc.trans) {
    //     ierr=MatBuildNonzeroColumnIndicator(prc.P,&prc.d);CHKERRQ(ierr);
    // } else {
    //     ierr=MatBuildNonzeroRowIndicator(prc.P,&prc.d);CHKERRQ(ierr);
    // }
    // ierr=VecScale(prc.d,-1.0);CHKERRQ(ierr);
    // ierr=VecShift(prc.d,1.0);CHKERRQ(ierr);
    
    PetscScalar delta;
    
    Vec *Vs;
    Vec x = p;
    Vec w;
    Vec g;
    
    // the matrix H will be oriented by columns, with k+1 rows of k columns
    PetscScalar *H;
    PetscScalar *svd_sigmas;
    PetscScalar *svd_V;
    PetscScalar *svd_work = PETSC_NULL;
    int svd_lwork = -1;
    
    ierr=PetscMalloc(sizeof(PetscScalar)*(k)*(k),&svd_V);CHKERRQ(ierr);
    ierr=PetscMalloc(sizeof(PetscScalar)*(k),&svd_sigmas);CHKERRQ(ierr);
    ierr=PetscMalloc(sizeof(PetscScalar)*(k)*(k+1),&H);CHKERRQ(ierr);
    
    ierr=VecDuplicate(p,&g);CHKERRQ(ierr);
    ierr=VecDuplicate(p,&w);CHKERRQ(ierr);
    ierr=VecDuplicateVecs(p,k,&Vs);CHKERRQ(ierr);
    
    PetscLogStagePush(STAGE_COMPUTE);
    
    if (prc.default_v) {
        ierr=VecSet(x,1.0/(PetscScalar)prc.N);CHKERRQ(ierr);
    }
    else {
        ierr=VecCopy(prc.v,x);CHKERRQ(ierr);
    }
    
    for (PetscInt iter = 0; iter < prc.maxiter; iter++) 
    {
        PetscScalar alpha,beta;
        memset(H, 0, sizeof(PetscScalar)*(k)*(k+1)); 
        
        ierr=VecCopy(x,Vs[0]);CHKERRQ(ierr);
        
        // just use delta as a temp here because I can't pass NULL
        ierr=VecNormalize(Vs[0],&delta);CHKERRQ(ierr);
        
        ierr=PageRankMult(prc,Vs[0],w);CHKERRQ(ierr);
        
        ierr=VecTDot(Vs[0],w,&alpha);CHKERRQ(ierr);
        H[0] = alpha;
        ierr=VecWAXPY(g,-alpha,Vs[0],w);CHKERRQ(ierr);

        for (PetscInt j=0; j < k-1; j++) {
            ierr=VecCopy(g,Vs[j+1]);CHKERRQ(ierr);
            ierr=VecNormalize(Vs[j+1],&beta);CHKERRQ(ierr);
            // H(j+1,j) in a column oriented matrix with k+1 rows, k cols
            H[j+1 + j*(k+1)] = beta; 
            ierr=PageRankMult(prc,Vs[j+1],w);CHKERRQ(ierr);
            ierr=VecMTDot(w,j+2,Vs,&H[(j+1)*(k+1)]);CHKERRQ(ierr);
            ierr=VecCopy(w,g);CHKERRQ(ierr);
            // negate H
            for (PetscInt hj=0; hj < j+2;  hj++) {
                H[(j+1)*(k+1) + hj]*= -1.0;
            }
            ierr=VecMAXPY(g,j+2,&H[(j+1)*(k+1)],Vs);CHKERRQ(ierr);
            // unnegate H
            for (PetscInt hj=0; hj < j+2;  hj++) {
                H[(j+1)*(k+1) + hj]*= -1.0;
            }
        }
        // Chen's code doesn't use V(:,k+1), but just 
        // computes its norm.
        //ierr=VecCopy(g,Vs[k]);CHKERRQ(ierr);
        //ierr=VecNormalize(Vs[k],&H[(k-1)*(k+1)+k);CHKERRQ(ierr);
        ierr=VecNorm(g,NORM_2,&H[(k-1)*(k+1)+k]);CHKERRQ(ierr);
        
        // subtract 1 to the diagonal of the matrix
        for (PetscInt hi=0; hi < k; hi++) {
            H[hi + hi*(k+1)]-=1.0;
        }

        // compute SVD
        {
            char job_u = 'N';
            char job_v = 'A';
            int svd_m = k+1;
            int svd_n = k;
            
            int lda = svd_m;
            
            int ldvt = k;
            
            int info = 0;
            
            int temp_i = 1;
            PetscScalar temp_s = 1.0; 
            
            if (svd_work == PETSC_NULL) {
                PetscScalar svd_work_len = 0;
                svd_lwork = -1;
                // do a workspace query
                LAPACKgesvd_(&job_u, &job_v, &svd_m, &svd_n, H,
                    &lda, svd_sigmas, &temp_s, &temp_i,
                    svd_V, &ldvt, &svd_work_len, &svd_lwork, &info);
                ierr=PetscMalloc(sizeof(PetscScalar)*(PetscInt)(svd_work_len), &svd_work);
                    CHKERRQ(ierr);
                svd_lwork = (PetscInt)svd_work_len;
            }
            
            LAPACKgesvd_(&job_u, &job_v, &svd_m, &svd_n, H,
                &lda, svd_sigmas, &temp_s, &temp_i,
                svd_V, &ldvt, svd_work, &svd_lwork, &info);
            
            if (info < 0) {
                SETERRQ1(PETSC_ERR_ARG_BADPTR, "SVD failed with info = %i\n", info);
            }
            else if (info > 0) {
                SETERRQ(PETSC_ERR_CONV_FAILED, "SVD failed on the Arnoldi matrix H");
            }
            
            // transpose V
            for (PetscInt hi = 0; hi < k; hi++) {
                for (PetscInt hj = hi+1; hj < k; hj++) {
                    PetscScalar temp = svd_V[hi + hj*(k)];
                    svd_V[hi + hj*k] = svd_V[hj + hi*k];
                    svd_V[hj + hi*k] = temp;
                }
            }
        }
        
        // update x
        ierr=VecSet(x,0.0);CHKERRQ(ierr);
        ierr=VecMAXPY(x,k,&svd_V[(k-1)*k],Vs);
        
        // check convergence norm(P*x - x,1)/norm(x,1)
        ierr=PageRankMult(prc,x,w);CHKERRQ(ierr);
        // compute w = w - x
        PetscLogStagePush(STAGE_EVALUATE);
        ierr=VecAXPY(w,-1.0,x);CHKERRQ(ierr);
        ierr=VecNorm(w,NORM_1,&delta);CHKERRQ(ierr);
        ierr=VecNorm(x,NORM_1,&beta);CHKERRQ(ierr);
        delta = delta/beta;
        PetscPrintf(prc.comm,"%4i  %10.3e\n", iter+1, delta);
        PetscLogStagePop();
       
        if (delta < prc.tol) {
            break;
        }
    }
    
    PetscLogStagePop();

    ierr=PetscFree(svd_V);
    ierr=PetscFree(svd_sigmas);
    ierr=PetscFree(svd_work);
    ierr=PetscFree(H);
    
    ierr=VecDestroyVecs(Vs, k);
    ierr=VecDestroy(g);
    ierr=VecDestroy(w);
                
    return (MPI_SUCCESS);
}

/**
 * Implement the PageRank multiplication operation on the full
 * matrix 
 * 
 * y = M(a)*x = a*P'*x + a*(d'*x)*v + (1-a)*(e'*x)*v
 * 
 * The multiplication is implemented by implicitly constructing
 * the dangling vector as d'*x = e'*x - e'*P*x.  This operation
 * take an identical number of flops because the quantity
 * e'*x is required for the other multiplication.
 */
#undef __FUNCT__
#define __FUNCT__ "PageRankMult"
PetscErrorCode PageRankMult(PageRankContext prc, Vec x, Vec y)
{
    //y = a*Pt*x + (a*(d'*x))*v + (1-a)*sum(x)*v;
    PetscErrorCode ierr;
    PetscScalar dtx;
    PetscScalar etx,etPtx;
    
    ierr=VecSum(x,&etx);CHKERRQ(ierr);
    if (prc.trans) {
        ierr=MatMult(prc.P,x,y);CHKERRQ(ierr);
    }
    else {
        ierr=MatMultTranspose(prc.P,x,y);CHKERRQ(ierr);
    }
    // TODO implement a routine to scale and sum y simultaneously
    ierr=VecSum(y,&etPtx);CHKERRQ(ierr);
    ierr=VecScale(y,prc.alpha);CHKERRQ(ierr);
    dtx = etx - etPtx;
    if (prc.default_v) {
        ierr=VecShift(y,(prc.alpha*dtx + (1-prc.alpha)*etx)/(PetscScalar)prc.N);CHKERRQ(ierr);
    }
    else {
        ierr=VecAXPY(y,prc.alpha*dtx + (1-prc.alpha)*etx,prc.v);CHKERRQ(ierr);
    }
    
    return (MPI_SUCCESS);
}

/**
 * This function computes a PageRank product with the adjustment from the
 * dangling node vector to compute a fully stochastic matrix-vector product.
 * 
 * Mathematically, it computes y = P'*x + (d'*x)*v.  Internally, it computes
 * s = e'*x, y = P'*x, d'*x = s - sum(y), so that it does not use 
 * the dangling vector.  This requires an extra n flops to implement, but
 * does not require the dangling vector.
 * 
 * We choose between the algorithms based on if prc.d exists or not.
 * 
 * @param prc the PageRank context
 * @param x the right hand side vector 
 * @param y the output (left hand side) vector
 */
 
#undef __FUNCT__
#define __FUNCT__ "PageRankDanglingMult"
PetscErrorCode PageRankDanglingMult(PageRankContext prc, Vec x, Vec y)
{
    
    PetscErrorCode ierr;
    PetscScalar dtx,etx,etPtx;
    
    if (prc.d) 
    {
        ierr=VecTDot(prc.d,x,&dtx);CHKERRQ(ierr);
        if (prc.trans) {
            ierr=MatMult(prc.P,x,y);CHKERRQ(ierr);
        }
        else {
            ierr=MatMultTranspose(prc.P,x,y);CHKERRQ(ierr);
        }
    } 
    else
    {
        // implement the implicit multiplication operation
        ierr=VecSum(x,&etx);CHKERRQ(ierr);
        if (prc.trans) {
            ierr=MatMult(prc.P,x,y);CHKERRQ(ierr);
        }
        else {
            ierr=MatMultTranspose(prc.P,x,y);CHKERRQ(ierr);
        }
        ierr=VecSum(y,&etPtx);CHKERRQ(ierr);
        dtx = etx - etPtx;

    }
    
    if (prc.default_v) {
        ierr=VecShift(y,(dtx)/(PetscScalar)prc.N);CHKERRQ(ierr);
    }
    else {
        ierr=VecAXPY(y,dtx,prc.v);CHKERRQ(ierr);
    }
    
    
    return (MPI_SUCCESS);
}
