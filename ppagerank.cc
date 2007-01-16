/*
 * David Gleich
 * 13 December 2006
 * Copyright, Stanford University
 */
 
/**
 * @file ppagerank.cc
 * The driver file for a distributed memory implementation of 
 * pagerank.
 */
 
#include "petsc.h"
#include "petscmat.h"
#include "petscmath.h"

#include "petsc_util.h"



const static int version_major = 0;
const static int version_minor = 0;

// function prototypes

PetscErrorCode WriteHeader();
PetscErrorCode WriteSimpleMatrixStats(const char* filename, Mat A);
PetscErrorCode ComputePageRank(Mat A);
PetscErrorCode MatLoadPickType(MPI_Comm comm, const char* filename, Mat *a, const char* filetypehint);

static char help[] = 
"usage: ppagerank -m <filename> [options]\n"
"\n"
"Data\n"
"  -m <filename>    (REQUIRED) Load the matrix given by filename\n"
"\n"
"PageRank parameters\n"
"  -alpha <float>   Set the value of the pagerank alpha parameter\n"
"                   default = 0.85\n"
"  -pvec <filename> Set the personalization vector\n"
"                   default = (1/n) for each entry\n"
"  -alg <algorithm> Set the algorithm to compute PageRank\n"
"                   default = power"
"\n"
"Additional options\n"   
"  -noout           Do not write any output information\n"
"  -trans           The input matrix is transposed\n"
"\n"
"Matrix Loading options\n"
"  -matload_root_nz_bufsize <int>     The nonzero buffer size to read and\n"
"                                     send non-zero values to other procs\n"
"                                     default = 2^25 (33554432)\n"
"  -matload_redistribute              A binary switch indicating that the \n"
"                                     matrix will be redistributed to balance\n"
"                                     non-zeros and rows among processors\n"
"  -matload_redistribute_wnnz <int>   The weight of each non-zero in the\n"
"                                     distribution.\n"
"                                     default = 1\n"
"  -matload_redistribute_wrows <int>  The weight of each row in the\n"
"                                     distribution.\n"
"                                     default = 1\n"
"";         
 
// add code to get correct debug reports 
#undef __FUNCT__
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    
    // parse the options
    ierr=PetscInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);

    if (argc < 2) {
        PetscPrintf(PETSC_COMM_WORLD, help);
        PetscFinalize();
        return (-1);
    }
    
    PetscTruth option_flag;
    
//    ierr=PetscOptionsHasName(PETSC_NULL,"-help",&option_flag);CHKERRQ(ierr);
//    if (option_flag) {
//        PetscFinalize();
//        return (-1);
//    }
    
    //
    // begin options parsing
    //
 
    // get the matrix filename
    char matrix_filename[PETSC_MAX_PATH_LEN];
    ierr=PetscOptionsGetString(PETSC_NULL,"-m",matrix_filename,PETSC_MAX_PATH_LEN,&option_flag);
        CHKERRQ(ierr); 
    if (!option_flag) {
        PetscPrintf(PETSC_COMM_WORLD,"\nOptions error: no matrix file specified!\n\n");
        PetscFinalize();
        return (-1);
    }
    
    PetscTruth script;
    ierr=PetscOptionsHasName(PETSC_NULL,"-script",&script);
    
    //
    // end options parsing
    //
       
    WriteHeader();
    
    Mat A;
    //ierr=MatLoadBSMAT(PETSC_COMM_WORLD,matrix_filename,&A);CHKERRQ(ierr);
    //ierr=MatLoadBVGraph(PETSC_COMM_WORLD,matrix_filename,&A);CHKERRQ(ierr);
    ierr=MatLoadPickType(PETSC_COMM_WORLD,matrix_filename,&A,NULL);CHKERRQ(ierr);
    
    WriteSimpleMatrixStats(matrix_filename, A);
    
    if (script) {
        // make sure there are no options left
        // PetscOptionsLeft();
        
        // PetscOptionsCreate();
        // PetscOptionsInsert(argc,argv)   
    } else {
        ierr=ComputePageRank(A); CHKERRQ(ierr);
    }
        
    PetscFinalize();
    
    return (0); 
} 



#undef __FUNCT__
#define __FUNCT__ "WriteHeader" 
PetscErrorCode WriteHeader(void)
{
    char name[MPI_MAX_PROCESSOR_NAME+1];
    int namelen;
    PetscErrorCode ierr;
    int rank, size;
    
    // print off the 
    for (int i=0; i<60; i++) { PetscPrintf(PETSC_COMM_WORLD, "%c", '='); }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    
    PetscPrintf(PETSC_COMM_WORLD, "ppagerank %i.%i\n\n", version_major, version_minor);
    PetscPrintf(PETSC_COMM_WORLD, "David Gleich\n");
    PetscPrintf(PETSC_COMM_WORLD, "Copyright, 2006\n");
    
    for (int i=0; i<60; i++) { PetscPrintf(PETSC_COMM_WORLD, "%c", '='); }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    
    ierr=MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
    ierr=MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    ierr=MPI_Get_processor_name(name, &namelen);
    
    PetscPrintf(PETSC_COMM_WORLD, "nprocs = %i\n", size);
    
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%3i] %s running...\n", rank, name);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
    
    ierr=MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    
    return (MPI_SUCCESS);
}		
 
/**
 * Output a set of simple matrix statistics.
 * 1) num rows, num cols, num non-zeros
 * 2) min/max rows/proc
 * 3) min/max cols/proc
 * 4) min/max off diag columns/proc
 * 
 * @param filename the matrix filename
 * @param A the distributed memory Petsc matrix
 */
#undef __FUNCT__
#define __FUNCT__ "WriteSimpleMatrixStats"
PetscErrorCode WriteSimpleMatrixStats(const char* filename, Mat A)
{
    MPI_Comm comm;
    PetscErrorCode ierr;
    
    PetscInt m,n;
    PetscInt ml,nl;
    
    PetscObjectGetComm((PetscObject)A,&comm);
    
    MatGetSize(A,&m,&n);
    MatGetLocalSize(A,&ml,&nl);
    
    PetscInt max_local_rows, min_local_rows;
    PetscInt max_local_columns, min_local_columns;
    
    ierr=MPI_Reduce(&ml,&max_local_rows,1,MPI_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr=MPI_Reduce(&ml,&min_local_rows,1,MPI_INT,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr=MPI_Reduce(&nl,&max_local_columns,1,MPI_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr=MPI_Reduce(&nl,&min_local_columns,1,MPI_INT,MPI_MIN,0,comm);CHKERRQ(ierr);
    
    long long int total_nz = 0;
    PetscInt local_nz = 0;
    ierr=MatGetNonzeroCount(A,&total_nz, &local_nz);
    
    PetscInt max_local_nz,min_local_nz;
    
    ierr=MPI_Reduce(&local_nz,&max_local_nz,1,MPI_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr=MPI_Reduce(&local_nz,&min_local_nz,1,MPI_INT,MPI_MIN,0,comm);CHKERRQ(ierr);

    PetscScalar mat_norm_1,mat_norm_inf;
    ierr=MatNorm(A,NORM_1,&mat_norm_1);CHKERRQ(ierr);
    ierr=MatNorm(A,NORM_INFINITY,&mat_norm_inf);CHKERRQ(ierr);

    PetscPrintf(comm,"\n");
    PetscPrintf(comm,"-----------------------------------------\n");        
    PetscPrintf(comm,"matrix statistics\n");
    PetscPrintf(comm,"-----------------------------------------\n");
    PetscPrintf(comm,"rows       =  %10i\n", m);
    PetscPrintf(comm,"columns    =  %10i\n", n);
    PetscPrintf(comm,"nnz        =  %10lli\n", total_nz);
    PetscPrintf(comm,"1-norm     =  %10g\n", mat_norm_1);
    PetscPrintf(comm,"inf-norm   =  %10g\n", mat_norm_inf);
    PetscPrintf(comm,"\n");
    PetscPrintf(comm,"              %10s  %10s\n", "min", "max");
    PetscPrintf(comm,"local rows =  %10i  %10i\n", min_local_rows, max_local_rows);
    PetscPrintf(comm,"local cols =  %10i  %10i\n", min_local_columns, max_local_columns);
    PetscPrintf(comm,"local nzs  =  %10i  %10i\n", min_local_nz, max_local_nz);  
    PetscPrintf(comm,"-----------------------------------------\n");
    PetscPrintf(comm,"\n");    
    return (MPI_SUCCESS);
}

/**
 * A quick wrapper to save writing lots of "if" statements for
 * matrices that are transposed.
 * 
 * @param A the matrix to create the vector for
 * @param v the output vector
 * @param trans a flag indicating if the matrix is transposed
 */
#undef __FUNCT__
#define __FUNCT__ "VecCreateForPossiblyTransposedMatrix"
PetscErrorCode VecCreateForPossiblyTransposedMatrix(Mat A, Vec *v, PetscTruth trans)
{
    if (trans) { return VecCreateForMatTranspose(A,v); }
    else { return VecCreateForMat(A,v); }
}

PetscErrorCode MatNormalizeForPageRank(Mat A,PetscTruth trans,Vec *d);

/**
 * Compute a PageRank vector for a PETSc Matrix A.
 * 
 * The ComputePageRank function loads all of its options from 
 * the command line.
 * 
 * @param A the matrix used for PageRank
 * @param transp a flag indicating if the matrix is transposed so that
 * the edge from page i to page j is in the ith column of the matrix
 * instead of the ith row of the matrix.
 * @return MPI_SUCCESS unless there was an error.
 */
#undef __FUNCT__
#define __FUNCT__ "ComputePageRank"
PetscErrorCode ComputePageRank(Mat A)
{
    PetscErrorCode ierr;
    PetscTruth flag;
    MPI_Comm comm;
    ierr=PetscObjectGetComm((PetscObject)A,&comm); CHKERRQ(ierr);
    
    PetscInt M,N;
    ierr=MatGetSize(A,&M,&N);CHKERRQ(ierr);
    
    // make sure the matrix is square
    if (M!=N) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"The matrix A was not square.");
    }
    
    // grab the current alpha
    PetscScalar alpha=0.85;
    ierr=PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,&flag);CHKERRQ(ierr);
    
    PetscTruth trans, noout;
    ierr=PetscOptionsHasName(PETSC_NULL,"-trans",&trans);CHKERRQ(ierr);
    ierr=PetscOptionsHasName(PETSC_NULL,"-noout",&noout);CHKERRQ(ierr);
       
    PetscTruth default_v = PETSC_TRUE;
    char pvec_filename[PETSC_MAX_PATH_LEN];
    ierr=PetscOptionsGetString(PETSC_NULL,"-pvec",pvec_filename,PETSC_MAX_PATH_LEN,&flag);
        CHKERRQ(ierr);
    default_v = (PetscTruth)!flag;
        
    PetscTruth require_d = PETSC_FALSE;
    
    Vec v;
    if (!default_v) 
    {
        // they are not using the default vector, so we cannot optimize
        // for that case
        ierr=VecCreateForPossiblyTransposedMatrix(A,&v,trans);CHKERRQ(ierr);

        // load their vector
        
        // TODO check for sparse vector input and load that more efficiently
        PetscViewer viewer;
        ierr=PetscViewerBinaryOpen(comm,pvec_filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr=VecLoad(viewer,PETSC_NULL,&v);CHKERRQ(ierr);
    }
    
    // normalize Matrix
    Vec d;
    ierr=MatNormalizeForPageRank(A,trans,require_d ? &d : PETSC_NULL);CHKERRQ(ierr);
    
    // TODO Check v for a probability distribution

    // quick implementation of the power method    
    PetscInt maxiter = 1000;
    PetscScalar tol = 1e-7;
    
    PetscPrintf(comm,"Computing PageRank...\n");
    PetscPrintf(comm,"alg = power\n");
    PetscPrintf(comm,"alpha = %f\n", alpha);
    PetscPrintf(comm,"maxiter = %i\n", maxiter);
    PetscPrintf(comm,"tol = %e\n", tol);
    PetscScalar mat_norm_1,mat_norm_inf;
    ierr=MatNorm(A,NORM_1,&mat_norm_1);CHKERRQ(ierr);
    ierr=MatNorm(A,NORM_INFINITY,&mat_norm_inf);CHKERRQ(ierr);
    PetscPrintf(comm,"||P||_1 = %f\n",mat_norm_1);
    PetscPrintf(comm,"||P||_inf = %f\n", mat_norm_inf);

    // note that we don't need to worry about the orientation of the matrix 
    // because it is square by assumption so the column vectors are row 
    // vectors.
    Vec x, y;
    ierr=VecCreateForMat(A,&x);CHKERRQ(ierr);
    ierr=VecDuplicate(x,&y);CHKERRQ(ierr);
    if (default_v) {
        ierr=VecSet(x,1.0/(PetscScalar)N);CHKERRQ(ierr);
    } else {
        ierr=VecCopy(v,x);CHKERRQ(ierr);
    }
    
    for (PetscInt iter = 0; iter < maxiter; iter++) 
    {
        if (trans) {
            ierr=MatMult(A,x,y);CHKERRQ(ierr);
        } else {
            ierr=MatMultTranspose(A,x,y);CHKERRQ(ierr);
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
        ierr=VecAYPX(x,-1.0,y);CHKERRQ(ierr);
        ierr=VecNorm(x,NORM_1,&delta);CHKERRQ(ierr);
        
        PetscPrintf(comm,"%4i  %10.3e\n", iter+1, delta);
        
        if (delta < tol) {
            break;
        }
        ierr=VecCopy(y,x);CHKERRQ(ierr);
    }
    
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
        ierr=VecCreateForMatTranspose(A,&col_align_vec);CHKERRQ(ierr);
        ierr=VecCreateForMat(A,&row_align_vec);CHKERRQ(ierr);
        
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

/**
 * Load a matrix file and determine the type from the extension
 * and possibly the filetype hint.
 * 
 * @param comm the MPI communicator
 * @param filename the matrix filename
 * @param A the future matrix
 * @param filetypehint the hint about the filetype
 */
#undef __FUNCT__
#define __FUNCT__ "MatLoadPickType"  
PetscErrorCode MatLoadPickType(MPI_Comm comm, const char* filename, Mat *A, const char* filetypehint)
{
    return MatLoadBVGraph(PETSC_COMM_WORLD,filename,A);
}

