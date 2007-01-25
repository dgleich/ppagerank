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
#include "petscoptions.h"
#include "petscmat.h"
#include "petscmath.h"

#include "petsc_util.h"
#include "ppagerank.h"

#define PPAGERANK_STAGES_DECLARE
#include "ppagerank_stages.h"

#include <string>
#include <vector>
#include <util/string.h>
#include <util/file.h>
#include <util/command_line.hpp>

const static int version_major = 0;
const static int version_minor = 0;

const static int options_line_size = 2048;

// function prototypes

PetscErrorCode WriteHeader();
PetscErrorCode WriteSimpleMatrixStats(const char* filename, Mat A);
PetscErrorCode SetupAndRunComputations(Mat A, PetscTruth script, PetscTruth trans);
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
    
    char matrix_filetype_hint[PETSC_MAX_PATH_LEN] = {};
    ierr=PetscOptionsGetString(PETSC_NULL,"-mhint",matrix_filetype_hint,PETSC_MAX_PATH_LEN,&option_flag);
        CHKERRQ(ierr);
    
    
    PetscTruth script=PETSC_FALSE;
    ierr=PetscOptionsHasName(PETSC_NULL,"-script",&script);
    
    PetscTruth trans = PETSC_FALSE;
    ierr=PetscOptionsHasName(PETSC_NULL,"-trans",&trans);CHKERRQ(ierr);
    
    //
    // end options parsing
    //
      
    RegisterStages(); 
    WriteHeader();
    
    PetscLogStagePush(STAGE_LOAD);
    Mat A;
    ierr=MatLoadPickType(PETSC_COMM_WORLD,matrix_filename,&A,matrix_filetype_hint);CHKERRQ(ierr);
    PetscLogStagePop();
    
    WriteSimpleMatrixStats(matrix_filename, A);
    
    SetupAndRunComputations(A,script,trans);
    if (script) { 
        // if these options are specified, we don't 
        // care about them at at this point.
        PetscOptionsClearValue("-m");
        PetscOptionsClearValue("-mhint");
        PetscOptionsClearValue("-script");
        PetscOptionsClearValue("-trans");
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
    bool hinted = false;
    {
        size_t len;
        PetscStrlen(filetypehint,&len);
        hinted = len > 0;
    }
    if (hinted)
    {
        PetscTruth flg;
        PetscStrncmp(filetypehint,"bsmat",PETSC_MAX_PATH_LEN,&flg);
        if (flg) {
            return MatLoadBSMAT(comm,filename,A);
        }
        
        PetscStrncmp(filetypehint,"bvgraph",PETSC_MAX_PATH_LEN,&flg);
        if (flg) {
            return MatLoadBVGraph(comm,filename,A);
        }
        
        PetscStrncmp(filetypehint,"cluto",PETSC_MAX_PATH_LEN,&flg);
        if (flg) {
        }
        
        PetscStrncmp(filetypehint,"smat",PETSC_MAX_PATH_LEN,&flg);
        if (flg) {
        }
        
        PetscStrncmp(filetypehint,"graph",PETSC_MAX_PATH_LEN,&flg);
        if (flg) {
        }
    }
    
    // get the extension
    std::string ext = util::split_filename(std::string(filename)).second;
    bool gzipped = false;
    if (ext.compare("gz") == 0) {
        gzipped = true;
        std::string no_gz_on_file = util::split_filename(std::string(filename)).first;
        ext = util::split_filename(no_gz_on_file).second;
    }
    
    if (ext.compare("bsmat") == 0) {
        return MatLoadBSMAT(comm,filename,A);
    }
    else if (ext.compare("smat") == 0) {
    }
    else if (ext.compare("graph") == 0) {
        // now try and determine which type of file it is
        if (!gzipped) {
            // a binary non-gzipped file must be a BVGraph
            util::filetypes ft = util::guess_filetype(filename);
            if (ft == util::filetype_binary) {
                return MatLoadBVGraph(comm,filename,A);
            }
        } 
    }
    
    if (hinted) { 
        PetscPrintf(comm,
        "Your hint, %s, was not used and the matrix time was not obvious\n"
        "from the file extension.  This either means the hint was misspelled\n"
        "or support for your matrix type isn't in the program yet.\n",
        filetypehint);
    }
    else {
        PetscPrintf(comm,
        "The matrix filename, %s, was not sufficient to determine the\n"
        "type of file.  Try using -mhint to give a hint about the filetype.\n",
        filename);
    }
    
    return (MPI_ERR_OTHER);
}

/**
 * Setup and run the computations.  This includes preprocessing
 * the data and handling the input if there is a run script.
 * 
 * If script is true, then this function will destroy any usage
 * data about the options and PetscOptionsLeft() will not 
 * function correctly.
 * 
 * @param A the raw data matrix A
 * @param script the parameter to switch on script mode
 * @param trans true if the matrix A was loaded in a transposed fashion
 */ 
#undef __FUNCT__
#define __FUNCT__ "SetupAndRunComputations"
PetscErrorCode SetupAndRunComputations(Mat A, PetscTruth script, PetscTruth trans)
{
    PetscErrorCode ierr;
    MPI_Comm comm;
    
    PetscObjectGetComm((PetscObject)A,&comm);
    
    // grab all the script lines at the start
    std::vector<std::string> script_lines;
    if (script) {
        // create a variable for the line
        char line[options_line_size] = {0};
        
        int eof;
        ierr=PetscSynchronizedFEof(PETSC_COMM_WORLD, stdin, &eof);CHKERRQ(ierr);
        while (!eof) {
            // read from stdin
            memset(line,0,sizeof(options_line_size)*sizeof(char));
            ierr=PetscSynchronizedFGets(PETSC_COMM_WORLD, stdin, 
                options_line_size, line);
                CHKERRQ(ierr);
            std::string linestr(line);
            // trim the string
            linestr.erase(0,linestr.find_first_not_of( "\t\n\r"));
            linestr.erase(linestr.find_last_not_of( "\t\n\r")+1);
            // only add non-null lines
            if (linestr.size() > 0) {
                script_lines.push_back(linestr);
            }
            ierr=PetscSynchronizedFEof(PETSC_COMM_WORLD, stdin, &eof);CHKERRQ(ierr);
        }
        
        PetscPrintf(comm,"\n");
        PetscPrintf(comm,"-----------------------------------------\n");        
        PetscPrintf(comm,"script options\n");
        PetscPrintf(comm,"-----------------------------------------\n");
        PetscPrintf(comm,"\n");
        for(unsigned int runindex = 0; runindex < script_lines.size(); ++runindex) {
            PetscPrintf(comm,"[%3i] %s\n", runindex+1, script_lines[runindex].c_str());
        }
        PetscPrintf(comm,"-----------------------------------------\n");
        PetscPrintf(comm,"\n");
    }
    
    
    // normalize the matrix
    ierr=MatNormalizeForPageRank(A,trans,PETSC_NULL);CHKERRQ(ierr);
    
    if (script) 
    {
        // get the program name
        char progname[PETSC_MAX_PATH_LEN];
        PetscGetProgramName(progname,PETSC_MAX_PATH_LEN);
        
        int margc;
        char **margv;
        ierr=PetscGetArgs(&margc,&margv);CHKERRQ(ierr);
        
        // make sure there are no options left
        PetscOptionsLeft();
        
        // remove all the options at this point
        PetscOptionsDestroy();
        
        for(unsigned int runindex = 0; runindex < script_lines.size(); ++runindex) {
             
            // parse to args
            char **sargv;
            int sargc;
            std::vector<std::string> args;
            util::split_command_line(script_lines[runindex],args);
  
            // add the argument which is the command name
            args.insert(args.begin(),std::string(margv[0]));
            util::args_to_c_argc_argv(args,sargc,sargv);
            
            // initialize a new set of options and add ours
            PetscOptionsCreate();
            PetscSetProgramName(progname);
            PetscOptionsInsert(&sargc,&sargv,(char*)0);
            
            ierr=ComputePageRank(A,trans); CHKERRQ(ierr);
            
            // make sure there are no options left
            PetscOptionsLeft();
            // remove all the options at this point
            PetscOptionsDestroy();
            
            // free the memory
            delete[] sargv[0];
            delete[] sargv;
        }
        
        // restore the options database
        PetscOptionsCreate();
        PetscSetProgramName(progname);
        PetscOptionsInsert(&margc, &margv,(char*)0);
    } else {
        ierr=ComputePageRank(A,trans); CHKERRQ(ierr);
    }
    
    return (MPI_SUCCESS);
}

