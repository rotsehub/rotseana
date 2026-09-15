#!/bin/tcsh

# ============================================================
# ROTSE IDL path setup
#
# This script configures IDL to use:
#
#   1. IDLAstro
#   2. ROTSE IDL routines from this GitHub checkout
#   3. Any pre-existing IDL libraries
#
# Default IDLAstro installation:
#   /home/smurotse/products/IDLAstro
#
# If IDLAstro is installed elsewhere:
#
#   source setup_idl_paths.csh /path/to/IDLAstro
#
# Usage:
#
#   cd /path/to/rotseana
#   source setup_idl_paths.csh
#
# ============================================================


# ------------------------------------------------------------
# Locate the downloaded rotseana repository
# ------------------------------------------------------------

set repo_root = `git rev-parse --show-toplevel`

if ($status != 0) then
    echo ""
    echo "ERROR: Run this script from inside the rotseana repository."
    echo ""
    return
endif

setenv ROTSEANA_HOME "$repo_root"


# ------------------------------------------------------------
# Set IDLAstro location
# ------------------------------------------------------------

# Default installation currently used at SMU
set default_idlastro = "/home/smurotse/products/IDLAstro"

# If the user supplies a path, use that instead.
if ($#argv >= 1) then
    setenv IDLASTRO_HOME "$argv[1]"
else if ($?IDLASTRO_HOME) then
    # Keep an IDLASTRO_HOME already defined by the user.
else
    setenv IDLASTRO_HOME "$default_idlastro"
endif


# ------------------------------------------------------------
# Check required directories
# ------------------------------------------------------------

if (! -d "$ROTSEANA_HOME/idl/umrotse") then
    echo ""
    echo "ERROR: Cannot find the GitHub ROTSE IDL directory:"
    echo "  $ROTSEANA_HOME/idl/umrotse"
    echo ""
    return
endif

if (! -d "$IDLASTRO_HOME/pro") then
    echo ""
    echo "ERROR: IDLAstro was not found at:"
    echo "  $IDLASTRO_HOME"
    echo ""
    echo "If IDLAstro is installed somewhere else, use:"
    echo ""
    echo "  source setup_idl_paths.csh /path/to/IDLAstro"
    echo ""
    return
endif


# ------------------------------------------------------------
# Preserve existing IDL libraries
# ------------------------------------------------------------

if ($?IDL_PATH) then
    set old_idl_path = "$IDL_PATH"
else
    set old_idl_path = "<IDL_DEFAULT>"
endif


# ------------------------------------------------------------
# Configure IDL_PATH
#
# IDLAstro is searched first.
#
# The idl/umrotse directory from THIS GitHub checkout is placed
# before any pre-existing ROTSE installation. Therefore routines
# such as find_burst.pro are taken from the downloaded repository.
# ------------------------------------------------------------

setenv IDL_PATH "+${IDLASTRO_HOME}:+${ROTSEANA_HOME}/idl/umrotse:${old_idl_path}"


# ------------------------------------------------------------
# Report configuration
# ------------------------------------------------------------

echo ""
echo "ROTSE IDL environment configured."
echo ""

echo "GitHub ROTSE IDL:"
echo "  $ROTSEANA_HOME/idl/umrotse"

echo ""
echo "IDLAstro:"
echo "  $IDLASTRO_HOME"

echo ""
echo "The GitHub ROTSE routines are placed before"
echo "pre-existing ROTSE libraries in IDL_PATH."

echo ""
echo "Start IDL with:"
echo "  idl"
echo ""
