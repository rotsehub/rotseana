# IDL find_burst

This directory contains the IDL implementation of the ROTSE `find_burst`
analysis and its findburst-specific helper routines.

## Included routines

- `find_burst.pro` - main find_burst routine
- `filter_obs.pro` - blind candidate filtering
- `lightcurve.pro` - light-curve variability calculations
- `make_rotse_name.pro` - ROTSE source-name generation
- `make_var_struct.pro` - variable structure definition
- `print_var.pro` - historical/output helper

## Current operational source

The current operational `find_burst.pro` on rotse14 is:

`/home/smurotse/idl.lib/find_burst.pro`

This version includes the validated fixes for blind-search RA/DEC output
and coordinate no-match handling.

## External IDL dependencies

The current environment also uses shared routines maintained elsewhere:

- `set_flags.pro`
- `check_flags.pro`
- `conv2deg.pro`
- `ivalue.pro`
- `lcplot2.pro`

These shared utilities are not duplicated in this directory.
