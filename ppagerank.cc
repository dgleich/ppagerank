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

#include "zlib.h"

const static int version_major = 0;
const static int version_minor = 0;

// function prototypes

PetscErrorCode WriteHeader();
PetscErrorCode MatLoadBSMAT(MPI_Comm comm,const char* filename, Mat *newmat);
void WriteSimpleMatrixStats(const char* filename, Mat A);

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
    
    WriteHeader();
    
    // load the matrix filename
 
    char matrix_filename[PETSC_MAX_PATH_LEN];
    ierr=PetscOptionsGetString(PETSC_NULL,"-m",matrix_filename,PETSC_MAX_PATH_LEN,&option_flag);
        CHKERRQ(ierr); 
    if (!option_flag) {
        PetscPrintf(PETSC_COMM_WORLD,"\nOptions error: no matrix file specified!\n\n");
        PetscFinalize();
        return (-1);
    }
    
    Mat A;
    ierr=MatLoadBSMAT(PETSC_COMM_WORLD,matrix_filename,&A);CHKERRQ(ierr);
    
    WriteSimpleMatrixStats(matrix_filename, A);
    
    
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
    
    return (MPI_SUCCESS);
}

// include extra file manipulation operators 
#include <util/file.hpp>
#include <util/string.hpp>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// declaration of MatCreateMPIAIJWithArrays
//EXTERN PetscErrorCode  MatCreateMPIAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);

#include <vector>

void RedistributeRows(MPI_Comm comm,
        PetscInt M, PetscInt m, 
        PetscInt* rowners,
        PetscInt* ourlens,
        PetscInt wrows, PetscInt wnnz);

/**
 * MatLoadBSMAT loads a binary sparse matrix to a distributed memory
 * Petsc MPIAIJ matrix.
 * 
 * The algorithm proceeds as follows, an action of r designates something
 * that occurs only on the root processor
 *
 * <pre> 
 *   initialization
 * r test that files exist
 * r open files
 * r send size
 *   receive size
 *   partition matrix equally amongst processors
 * r read degree file and send degree file to each processor
 *   receive degree file
 *   redistribute the matrix according to row/edge balance
 * r read through file a bit at a time, and prepare send buffers for other processors
 *   receive data from root until all data received
 * </pre>
 *   
 * @param comm the MPI communicator loading the matrix
 * @param filename the filename of the matrix file
 * @param newmat a pointer to the structure that will hold the new matrix
 * @return MPI_SUCCESS on success!
 */
#undef __FUNCT__
#define __FUNCT__ "MatLoadBSMAT" 
PetscErrorCode MatLoadBSMAT(MPI_Comm comm_in, const char* filename, Mat *newmat)
{
    PetscMPIInt rank,size;
	PetscMPIInt tag;
	MPI_Status status;
    MPI_Comm   comm;
    PetscErrorCode ierr;

    ierr=PetscCommDuplicate(comm_in,&comm,&tag);CHKERRQ(ierr);
        
    // get the rank and size
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // allocate a variable to indicate the root processor
    bool root = !rank;
    
    // allocate all the file descriptors
    bool gzipped_mat=false, gzipped_degs=false;
    int fd_mat, fd_degs;
    gzFile gfd_mat, gfd_degs;
    bool matfile_error=false,degsfile_error=false;
    

    if (root) {
        
        //
        // test if the files exist
        // open files
        // 
         
        // determine if the files are gzipped
        if (util::gzip_header(filename)) { gzipped_mat = true; }
        
        std::string filename_degs = std::string(filename) + ".degs";
        if (gzipped_mat) { 
            filename_degs = util::split_filename(std::string(filename)).first + ".degs";
        }
        
        bool mat_exists = util::file_exists(filename);
        bool degs_exists = util::file_exists(filename_degs);
        if (!degs_exists) {
            // try the gzipped file
            std::string filename_degs_gz = filename_degs + ".gz";
            PetscPrintf(comm, "... checking %s for degrees ...\n", 
                filename_degs_gz.c_str());
                
            // test if the gzipped file exists
            degs_exists = util::file_exists(filename_degs_gz);
            
            if (degs_exists) {
                //  make sure the file is really a gzipped file
                gzipped_degs = util::gzip_header(filename_degs_gz);
                if (!gzipped_degs) {
                    SETERRQ1(PETSC_ERR_FILE_OPEN,
                        "The degree file %s is not a valid gzip file\n",
                        filename_degs_gz.c_str());
                }
                // save the filename if it does exist
                filename_degs = filename_degs_gz;
            }
        }
        
        if (!mat_exists) { 
            SETERRQ1(PETSC_ERR_FILE_OPEN,
                "The matrix file %s does not exist",filename); 
        }
        if (!degs_exists) { 
            SETERRQ2(PETSC_ERR_FILE_OPEN,
                "Cannot find degree file %s or %s.gz",
                filename_degs.c_str(), filename_degs.c_str());
        }
        
        PetscPrintf(comm,"matrix filename = %s\n", filename);
        PetscPrintf(comm,"degree filename = %s\n", filename_degs.c_str());
        
        if (gzipped_mat) {
            gfd_mat = gzopen(filename,"rb");
            matfile_error = !gfd_mat;
        } else {
            fd_mat = open(filename,O_RDONLY|O_LARGEFILE);
            matfile_error = fd_mat == -1;
        }
        
        if (gzipped_mat) {
            gfd_degs = gzopen(filename_degs.c_str(),"rb");
            degsfile_error = !gfd_degs;
        } else {
            fd_degs = open(filename_degs.c_str(),O_RDONLY|O_LARGEFILE);
            degsfile_error = fd_degs == -1;
        }
        
        // report any errors and terminate
        if (matfile_error) {
            SETERRQ1(PETSC_ERR_FILE_OPEN,
                "Error while opening %s\n", filename);
        }
        if (degsfile_error) {
            SETERRQ1(PETSC_ERR_FILE_OPEN,
                "Error while opening %s\n", filename_degs.c_str());
        }           
    }
        

    
    //
    // r send size
    //   receive size
    //
    
    PetscInt M,N;
    
    if (root) {
        // read size
        if (gzipped_mat) {
            gzread(gfd_mat, &M, sizeof(PetscInt)*1);
            gzread(gfd_mat, &N, sizeof(PetscInt)*1);
            gzseek(gfd_mat, sizeof(unsigned int), SEEK_CUR);
        }
        else {
            read(fd_mat, &M, sizeof(PetscInt)*1);
            read(fd_mat, &N, sizeof(PetscInt)*1);
            lseek(fd_mat, sizeof(unsigned int), SEEK_CUR);
        }
        
    }
   
    ierr=MPI_Bcast(&M,1,MPIU_INT,0,comm); CHKERRQ(ierr);
    ierr=MPI_Bcast(&N,1,MPIU_INT,0,comm); CHKERRQ(ierr);
    
    //
    // partition matrix equally amongst processors
    //
    
    PetscInt m,n;
    // determine ownership of all rows
    m = M/size + ((M%size)>rank);
    
    PetscInt *rowners;
    PetscMalloc((size+1)*sizeof(PetscInt),&rowners);
    ierr=MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
    
	// compute the rowners vector
    // processor i has rows rowners[i] to rowners[i+1]
	rowners[0] = 0;
	for (PetscInt i=2; i<=size; i++) {
		rowners[i] += rowners[i-1];
	}
	
	PetscInt rstart, rend;
	rstart = rowners[rank];
	rend = rowners[rank+1];
    
    // 
    // * r read degree file and send degree file to each processor
    // *   receive degree file
    //
	
	// ourlens is the length of each row in the local portion of the matrix
	// offlens is the length of each row in the non local portion of the matrix
	PetscInt *ourlens, *offlens;
	
	// allocate memory
	PetscMalloc2(m+1,PetscInt,&ourlens,m+1,PetscInt,&offlens);
	
	if (root) {
		// root readin the local part
		if (gzipped_degs) {
			gzread(gfd_degs, ourlens, sizeof(PetscInt)*m);
		}
		else {
			read(fd_degs, ourlens, sizeof(PetscInt)*m);
		}
		
		PetscInt mmax=rowners[1]-rowners[0];
		for (PetscInt i=1; i<size; i++) {
			mmax = PetscMax(mmax,rowners[i+1]-rowners[i]);
		}
		
		if (size > 1) { 
			std::vector<PetscInt> off_proc_degs(mmax);
			for (PetscInt i=1; i<size; i++) {
				if (gzipped_degs) {
					gzread(gfd_degs, &off_proc_degs[0], sizeof(PetscInt)*(rowners[i+1]-rowners[i]));
				}
				else {
					read(fd_degs, &off_proc_degs[0], sizeof(PetscInt)*(rowners[i+1]-rowners[i]));
				}
				MPI_Send(&off_proc_degs[0], rowners[i+1]-rowners[i], MPIU_INT, i, tag, comm);
			}
		}
	}
	else {
		// wait for data to arrive
		MPI_Recv(ourlens,m,MPIU_INT,0,tag,comm,&status);
	}
	
	//
	// redistribute the matrix according to row/edge balance
	//
	
	PetscTruth redistribute=PETSC_FALSE;
	PetscInt wrows=1, wnnz=1;
	
	PetscOptionsHasName(PETSC_NULL, "-matload_redistribute",&redistribute);	
	PetscOptionsGetInt(PETSC_NULL, "-matload_redistribute_wrows",&wrows, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-matload_redistribute_wnnz",&wnnz, PETSC_NULL);

	
	if (redistribute) {
	
		RedistributeRows(comm, M, m, rowners, ourlens, wrows, wnnz);
	
		PetscFree2(ourlens,offlens);

		m = rowners[rank+1] - rowners[rank];
		rstart = rowners[rank];
		rend = rowners[rank+1];


		// 
		// begin repeat of earlier block to read the degrees file
		//

    	// 
    	// * r read degree file and send degree file to each processor
    	// *   receive degree file
    	//

		// allocate memory
		PetscMalloc2(m+1,PetscInt,&ourlens,m+1,PetscInt,&offlens);

		if (root) {
			// root readin the local part
			if (gzipped_degs) {
				gzread(gfd_degs, ourlens, sizeof(PetscInt)*m);
			}
			else {
				read(fd_degs, ourlens, sizeof(PetscInt)*m);
			}

			PetscInt mmax=rowners[1]-rowners[0];
			for (PetscInt i=1; i<size; i++) {
				mmax = PetscMax(mmax,rowners[i+1]-rowners[i]);
			}

			if (size > 1) { 
				std::vector<PetscInt> off_proc_degs(mmax);
				for (PetscInt i=1; i<size; i++) {
					if (gzipped_degs) {
						gzread(gfd_degs, &off_proc_degs[0], sizeof(PetscInt)*(rowners[i+1]-rowners[i]));
					}
					else {
						read(fd_degs, &off_proc_degs[0], sizeof(PetscInt)*(rowners[i+1]-rowners[i]));
					}
					MPI_Send(&off_proc_degs[0], rowners[i+1]-rowners[i], MPIU_INT, i, tag, comm);
				}
			}
		}
		else {
			// wait for data to arrive
			MPI_Recv(ourlens,m,MPIU_INT,0,tag,comm,&status);
		}

		// 
		// end repeat of earlier block
		//
	}
	
    //
    // compute the local nz count for each processor and
    // compute the total nz count for all processors
    //
    
	// compute the nnz that will be local
	PetscInt local_nz=0;
	for (PetscInt i=0; i<m; i++) { local_nz += ourlens[i]; }
    
    PetscInt *procs_nz;
    PetscMalloc(sizeof(PetscInt)*size, &procs_nz);
    procs_nz[rank] = local_nz;
    
    PetscInfo1(PETSC_NULL," local_nz = %i\n",local_nz);
    
    // compute the total nnz
    ierr=MPI_Allgather(&local_nz,1,MPIU_INT,procs_nz,1,MPIU_INT,comm);CHKERRQ(ierr);
    long long int total_nz = 0;
    for (PetscInt i=0; i<size; i++) { total_nz += procs_nz[i]; }
    
    // 
    // r read through file a bit at a time, and prepare send buffers for other processors
    //   receive data from root until all data received
    //
	   
    
    // allocate a series of local buffers to store the data as it is received
    unsigned int *local_nz_i, *local_nz_j;
    double *local_nz_v;
    
    PetscMalloc((local_nz)*sizeof(unsigned int), &local_nz_i);
    PetscMalloc((local_nz)*sizeof(unsigned int), &local_nz_j);
    PetscMalloc((local_nz)*sizeof(double), &local_nz_v);
    
    // allocate a series of buffers for the Scatterv call
    PetscInt *sendcounts;
    PetscInt *displacements;
    
    PetscMalloc(sizeof(PetscInt)*size,&sendcounts);
    PetscMalloc(sizeof(PetscInt)*size,&displacements);
    
    // determine how much data the root will read from the 
    // matrix in each round
    PetscInt root_nz_bufsize = 1 << 25;
    PetscOptionsGetInt(PETSC_NULL,"-matload_root_nz_bufsize",&root_nz_bufsize,PETSC_NULL);
    
    // determine how many rounds of data exchange we have between
    // the root and all other processors
    long long send_rounds = total_nz / root_nz_bufsize + (total_nz%root_nz_bufsize > 0);
    
	if (root) {
        
        // this variable indicates the current position 
        PetscInt cur_nz=0;
        
        // the root processor reads data into a big block of 
        // root_nz_buf, 
        // and then divides the data into arrays root_nz_buf_i
        // root_nz_buf_j, and root_nz_buf_v such that
        // all the data going to each processor is adjacent in the
        // set of three arrays
        
        unsigned char* root_nz_buf;
        unsigned int *root_nz_buf_i,*root_nz_buf_j;
        double *root_nz_buf_v;
        PetscMalloc((sizeof(unsigned int)*2+sizeof(double))*root_nz_bufsize,&root_nz_buf);
        PetscMalloc(sizeof(unsigned int)*root_nz_bufsize,&root_nz_buf_i);
        PetscMalloc(sizeof(unsigned int)*root_nz_bufsize,&root_nz_buf_j);
        PetscMalloc(sizeof(double)*root_nz_bufsize,&root_nz_buf_v);
        
        long long int nzs_to_read = total_nz;
        
        while (send_rounds > 0) {
            // check if we are near the end of the file
            // and just read that amount
            size_t cur_nz_read = root_nz_bufsize;
            if (cur_nz_read > nzs_to_read) {
                cur_nz_read = nzs_to_read;
            }
            PetscInfo2(PETSC_NULL," reading %i non-zeros of %lli\n", cur_nz_read, nzs_to_read);
            // read data from the file
            if (gzipped_mat) {
                gzread(gfd_mat,root_nz_buf,(sizeof(unsigned int)*2+sizeof(double))*cur_nz_read);
            } else {
                read(fd_mat,root_nz_buf,(sizeof(unsigned int)*2+sizeof(double))*cur_nz_read);
            }
            nzs_to_read -= cur_nz_read;
            // parse through the data
            PetscMemzero(sendcounts,sizeof(PetscInt)*size);
            PetscMemzero(displacements,sizeof(PetscInt)*size);
            for (PetscInt i=0; i < cur_nz_read; i++) {
                unsigned int nzi = *((int*)(&root_nz_buf[i*(sizeof(unsigned int)*2+sizeof(double))]));
                unsigned int nzj = *((int*)(&root_nz_buf[sizeof(unsigned int)+i*(sizeof(unsigned int)*2+sizeof(double))]));
                
                if (nzi >= (unsigned int)M) {
                    // throw error on row
                    SETERRQ2(PETSC_ERR_FILE_UNEXPECTED,
                        "row=%i is out of range in non-zero %lli",
                        nzi, total_nz - nzs_to_read + (i+1));
                }
                if (nzj >= (unsigned int)N) {
                    // throw error on column
                    SETERRQ2(PETSC_ERR_FILE_UNEXPECTED,
                        "column=%i is out of range in non-zero %lli",
                        nzj, total_nz - nzs_to_read + (i+1));
                }
                
                for (PetscInt p=0; p<size; p++) {
                    if (nzi >= rowners[p] && nzi < rowners[p+1]) {
                        sendcounts[p]++;
                        break;
                    }
                }
            }

            // compute all the offsets were will store entries 
            // in the arrays to send to each other processor            
            PetscInt dspment = sendcounts[0];
            for (PetscInt p=1; p < size; p++) {
                displacements[p] = dspment;
                dspment += sendcounts[p];
                
                PetscInfo2(PETSC_NULL," sendcounts[%i] = %i\n", p, sendcounts[p]);
                PetscInfo2(PETSC_NULL," displacement[%i] = %i\n", p, displacements[p]);
            }
            PetscMemzero(sendcounts,sizeof(PetscInt)*size);
            
            
            
            PetscInfo(PETSC_NULL," test...\n");
            
            for (PetscInt i=0; i < cur_nz_read; i++) {
                unsigned int nzi = *((int*)(&root_nz_buf[i*(sizeof(unsigned int)*2+sizeof(double))]));
                unsigned int nzj = *((int*)(&root_nz_buf[sizeof(unsigned int)+i*(sizeof(unsigned int)*2+sizeof(double))]));
                double nzv = *((double*)(&root_nz_buf[2*sizeof(unsigned int)+i*(sizeof(unsigned int)*2+sizeof(double))]));
                
                PetscInt p=0;
                for (p=0; p<size-1; p++) {
                    if (nzi >= rowners[p] && nzi < rowners[p+1]) {
                        break;
                    }
                }
                
                root_nz_buf_i[displacements[p]+sendcounts[p]] = nzi;
                root_nz_buf_j[displacements[p]+sendcounts[p]] = nzj;
                root_nz_buf_v[displacements[p]+sendcounts[p]] = nzv;
                
                sendcounts[p]++;
            }
            
            PetscInfo(PETSC_NULL," test 2...\n");
                
            // now send
            MPI_Scatterv(root_nz_buf_i,sendcounts,displacements,MPI_INT,
                &local_nz_i[cur_nz],local_nz - cur_nz,MPI_INT,
                0,comm);
            MPI_Scatterv(root_nz_buf_j,sendcounts,displacements,MPI_INT,
                &local_nz_j[cur_nz],local_nz - cur_nz,MPI_INT,
                0,comm);
            MPI_Scatterv(root_nz_buf_v,sendcounts,displacements,MPI_DOUBLE,
                &local_nz_v[cur_nz],local_nz - cur_nz,MPI_DOUBLE,
                0,comm);
            cur_nz += sendcounts[rank];
            send_rounds--;
        }
        
        if (cur_nz != local_nz) {
            SETERRQ2(PETSC_ERR_FILE_UNEXPECTED,
                "processor %i received only %i nonzeros but expected %i\n",
                rank, cur_nz, local_nz);
        }
        
        // free all the memory
        PetscFree(root_nz_buf);
        PetscFree3(root_nz_buf_i, root_nz_buf_j, root_nz_buf_v);
        
	}
	else {
        PetscInt cur_nz=0;
        while (send_rounds > 0) {
            MPI_Scatterv(PETSC_NULL,sendcounts,displacements,MPI_INT,
                &local_nz_i[cur_nz],local_nz - cur_nz,MPI_INT,
                0,comm);
            MPI_Scatterv(PETSC_NULL,sendcounts,displacements,MPI_INT,
                &local_nz_j[cur_nz],local_nz - cur_nz,MPI_INT,
                0,comm);
            MPI_Scatterv(PETSC_NULL,sendcounts,displacements,MPI_DOUBLE,
                &local_nz_v[cur_nz],local_nz - cur_nz,MPI_DOUBLE,
                0,comm);
            cur_nz += sendcounts[rank];
            send_rounds--;
        }
        
        if (cur_nz != local_nz) {
            SETERRQ2(PETSC_ERR_FILE_UNEXPECTED,
                "processor %i received only %i nonzeros but expected %i\n",
                rank, cur_nz, local_nz);
        }
	}
    
    
    
    //
    // now, each processor has the set of non-zeros it will 
    // maintain, the remaining steps are all fairly independent
    // and involve assembling each matrix on the processor
    // 
	PetscInt *local_cols;
    PetscScalar *local_vals;
    
    // allocate memory
    PetscMalloc(sizeof(PetscInt)*local_nz, &local_cols);
    PetscMalloc(sizeof(PetscScalar)*local_nz, &local_vals);
    
    // we will use the offlens array to hold data about where to insert each element
    PetscMemzero(offlens, sizeof(PetscInt)*m);
    
    {
        offlens[0] = 0;
        PetscInt offset = ourlens[0];
        for (PetscInt i=1; i < m+1; i++) {
            offlens[i]=offset;
            offset += ourlens[i];
        }
    }
    for (PetscInt i=0; i < local_nz; i++) {
        PetscInt adjrow = local_nz_i[i]-rstart;
        local_cols[offlens[adjrow]] = local_nz_j[i];
        local_vals[offlens[adjrow]] = local_nz_v[i];   
        offlens[adjrow]++;
    }
    for (PetscInt i=m-1; i >= 0; i--) {
        offlens[i+1] = offlens[i];
    }
    offlens[0] = 0;
        
    ierr=PetscFree(local_nz_i);
    ierr=PetscFree(local_nz_j);
    ierr=PetscFree(local_nz_v);
    
    // we now have all data on the processor in CSR format.
    
    //MatCreateMPIIJWithArrays(comm,m,PETSC_DECIDE,M,N,
    //    offlens,local_cols,local_vals, &A);
    
    // set the number of columns
    if (M == N) {
        n = m;
    }
    else {
        n = N/size + ((N % size) > rank);
    }
    
    MatCreate(comm,newmat);
    MatSetSizes(*newmat,m,n,M,N);
    if (size > 1) {
        MatSetType(*newmat,MATMPIAIJ);
        MatMPIAIJSetPreallocationCSR(*newmat,offlens,local_cols,local_vals);
    }
    else {
        MatSetType(*newmat,MATSEQAIJ);
        MatSeqAIJSetPreallocationCSR(*newmat,offlens,local_cols,local_vals);
    }
    
    PetscFree2(ourlens,offlens);
	PetscFree(rowners);
	
	PetscFree(local_cols);
	PetscFree(local_vals);
    
    PetscFree(sendcounts);
    PetscFree(displacements);
    
    PetscCommDestroy(&comm);
        
    return (MPI_SUCCESS);
}

/**
 * Compute a new distribution of rows 
 */
void RedistributeRows(MPI_Comm comm,
		PetscInt M, PetscInt m, 
		PetscInt* rowners,
		PetscInt* ourlens,
		PetscInt wrows, PetscInt wnnz)
{
	PetscInfo(PETSC_NULL, "redistributing the rows of the matrix...");
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
#define __FUNCT__ "MatLoadBSMAT"
void WriteSimpleMatrixStats(const char* filename, Mat A)
{
    MPI_Comm comm;
    
    PetscInt m,n;
    PetscInt ml,nl;
    MatInfo global_info;
    MatInfo max_global_info;
    MatInfo local_info;
    
    /*MatGetInfo(A,MAT_GLOBAL,&global_info);
    MatGetInfo(A,MAT_GLOBLAL_MAX,&max_global_info);
    MatGetInfo(A,MAT_LOCAL,&local_info);*/
    
    PetscObjectGetComm((PetscObject)A,&comm);
    
    MatGetSize(A,&m,&n);
    MatGetLocalSize(A,&ml,&nl);
    
    PetscPrintf(comm,"matrix %s\n", filename);
    PetscPrintf(comm,"rows = %i\n", m);
    PetscPrintf(comm,"columns = %i\n", n);
}

