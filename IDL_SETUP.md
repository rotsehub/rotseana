# ROTSE IDL Setup

This repository contains the ROTSE IDL code under:

    idl/umrotse/

The `setup_idl_paths.csh` script configures IDL so that routines from
this GitHub checkout are used before older ROTSE installations that may
already exist on the system.

## 1. Clone the repository

    git clone https://github.com/rotsehub/rotseana.git
    cd rotseana

## 2. Configure the IDL paths

On the SMU ROTSE system, run:

    source setup_idl_paths.csh

The default IDLAstro installation is:

    /home/smurotse/products/IDLAstro

The ROTSE routines are automatically taken from the `idl/umrotse`
directory in the downloaded GitHub repository.

For example, `find_burst.pro` will be loaded from:

    <path-to-rotseana>/idl/umrotse/findburst/find_burst.pro

instead of an older local copy.

## 3. Use a different IDLAstro installation

If IDLAstro is installed somewhere else, specify its location:

    source setup_idl_paths.csh /path/to/IDLAstro

For example:

    source setup_idl_paths.csh /home/student/software/IDLAstro

## 4. Start IDL

    idl

## 5. Verify the ROTSE code being used

Inside IDL:

    resolve_routine, 'find_burst', /is_function
    print, (routine_info('find_burst', /source, /functions)).path

The output should point to:

    .../rotseana/idl/umrotse/findburst/find_burst.pro

## 6. Verify IDLAstro

Inside IDL:

    resolve_routine, 'sixty', /is_function
    print, (routine_info('sixty', /source, /functions)).path

On the SMU ROTSE system, the output should point to:

    /home/smurotse/products/IDLAstro/pro/sixty.pro

## Notes

- Older copies of `find_burst.pro` do not need to be deleted.
- The GitHub `idl/umrotse` tree is placed before pre-existing ROTSE
  libraries in the IDL search path.
- Run `source setup_idl_paths.csh` in each new terminal session before
  starting IDL.
