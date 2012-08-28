/*
 * David Gleich
 * 30 January 2007
 * Copyright, Stanford University
 */
 
/**
 * @file prvs_basic.c
 * A set of PETSC Functions to initialize the PRVS objects 
 */


#include "prvsimpl.h" 

PetscFList PRVSList = 0;
PetscCookie PRVS_COOKIE = 0;
PetscEvent PRVS_SetUp = 0, PRVS_Solve = 0;

#undef __FUNCT__
#define __FUNCT__ "PRVSInitializePackage"
PetscErrorCode PRVSInitializePackage(char* path)
{
    static PetscTruth initialized = PETSC_FALSE;
    char              logList[256];
    char             *className;
    PetscTruth        opt;
    PetscErrorCode    ierr;
    
    PetscFunctionBegin;
    if (initialized) { PetscFunctionReturn(0); }
    
    initialized = PETSC_TRUE;
    
    /* Register Classes */
    ierr=PetscLogRegister(&PRVS_COOKIE,"PageRank Vector Solver");CHKERRQ(ierr);
    
    /* Register Constructor */
    /* ierr=PRVSRegisterAll(path);CHKERRQ(ierr); */
    
    /* Register Events */
    ierr=PetscLogEventRegister(&PRVS_SetUp,"PRVSSetUp",PRVS_COOKIE);CHKERRQ(ierr);
    ierr=PetscLogEventRegister(&PRVS_Solve,"PRVSSolve",PRVS_COOKIE);CHKERRQ(ierr);
    
    /* Process Exclusions */
    ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
    if (opt) {
        ierr = PetscStrstr(logList,"prvs",&className);CHKERRQ(ierr);
        if (className) {
            ierr=PetscInfoDeactiveClass(PRVS_COOKIE); CHKERRQ(ierr);
        }
    }
    ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
    if (opt) {
        ierr = PetscStrstr(logList,"prvs",&className);CHKERRQ(ierr);
        if (className) {
            ierr=PetscLogEventDeactiveClass(PRVS_COOKIE); CHKERRQ(ierr);
        }
    }
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PRVSView"
PetscErrorCode PRVSView(PRVS prvs, PetscViewer viewer)
{
    PetscErrorCode   ierr;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(prvs,PRVS_COOKIE,1);
    if (!viewer) viewer = PETSC_VIEWER_STDOUT_(prvs->comm);
    PetscCheckSameComm(prvs,1,viewer,2);
    
    ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
    if (isascii) {
        PetscViewerASCIIPrintf(viewer,"PageRank Vector Solver\n");
    }
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PRVSPublish_Petsc"
static PetscErrorCode PRVSPublish_Petsc(PetscObject object)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PRVSCreate"
PetscErrorCode PRVSCreate(MPI_Comm comm, PRVS *outprvs)
{
    PetscErrorCode ierr;
    PRVS           prvs;
    
    PetscFunctionBegin;
    PetscValidPointer(outprvs,2);
    *outprvs = 0;
    
    PetscHeaderCreate(prvs,_p_PRVS,struct _PRVSOps,PRVS_COOKIE,-1,"PRVS",comm,PRVSDestroy,PRVSView);
    PetscLogObjectCreate(prvs);
    
    *outprvs = prvs;
    
    prvs->bops->publish = PRVSPublish_Petsc;
    ierr=PetscMemzero(prvs->ops,sizeof(struct _PRVSOps));CHKERRQ(ierr);
    
    /* initialize all the variables in the solver */
    prvs->type          = -1;
    
    ierr = PetscPublishAll(eps);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PRVSSetType"
PetscErrorCode PRVSSetType(PRVS eps,PRVSType type)
{
    PetscErrorCode ierr,(*r)(PRVS);
    PetscTruth match;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(prvs,PRVS_COOKIE,1);
    PetscValidCharPointer(type,2);
    
    ierr=PetscTypeCompare((PetscObject)prvs,type,&match);CHKERRQ(ierr);
    if (match) { PetscFunctionReturn(0); }
    
    if (prvs->data) {
        /* destroy old data */
        ierr = (*prvs->ops->destroy)(prvs); CHKERRQ(ierr);
        prvs->data = 0;
    }
    
    ierr = PetscFListFind(prvs->comm,PRVSList,type,(void (**)(void)) &r);CHKERRQ(ierr);
    if (!r) { SETERRQ1(1,"Unknown EPS type given: %s",type); }
    
    prvs->setup_called = 0;
    ierr = PetscMemzero(prvs->ops,sizeof(struct _PRVSOps));CHKERRQ(ierr);
    ierr = (*r)(prvs); CHKERRQ(ierr);
    
    ierr = PetscObjectChangeTypeName((PetscObject)prvs,type);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*@C
   PRVSGetType - Gets the PRVS type as a string from the EPSPRVS object.

   Not Collective

   Input Parameter:
.  prvs - the pagerank vector solver context 

   Output Parameter:
.  name - name of PRVS method 

   Level: intermediate

.seealso: PRVSGetType()
@*/
#undef __FUNCT__  
#define __FUNCT__ "PRVSGetType"
PetscErrorCode PRVSGetType(PRVS eps,PRVSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(prvs,PRVS_COOKIE,1);
  *type = prvs->type_name;
  PetscFunctionReturn(0);
}

/*MC
   EPSRegisterDynamic - Adds a method to the eigenproblem solver package.

   Synopsis:
   EPSRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(EPS))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   EPSRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   EPSRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     EPSSetType(eps,"my_solver")
   or at runtime via the option
$     -eps_type my_solver

   Level: advanced

   Environmental variables such as ${PETSC_ARCH}, ${SLEPC_DIR}, ${BOPT},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.seealso: EPSRegisterAll()

M*/

#undef __FUNCT__  
#define __FUNCT__ "EPSRegister"
PetscErrorCode EPSRegister(const char *sname,const char *path,const char *name,int (*function)(EPS))
{
  PetscErrorCode ierr;
  char           fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&EPSList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy"
/*@
   EPSDestroy - Destroys the EPS context.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSSolve()
@*/
PetscErrorCode EPSDestroy(EPS eps)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(prvs,PRVS_COOKIE,1);
    if (--prvs->refct > 0) PetscFunctionReturn(0);

    /* if memory was published with AMS then destroy it */
    ierr = PetscObjectDepublish(eps);CHKERRQ(ierr);

    ierr = STDestroy(eps->OP);CHKERRQ(ierr);

    if (eps->ops->destroy) {
        ierr = (*eps->ops->destroy)(eps); CHKERRQ(ierr);
    }

    PetscLogObjectDestroy(prvs);
    PetscHeaderDestroy(prvs);
    PetscFunctionReturn(0);
}

