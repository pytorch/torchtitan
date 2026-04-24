SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

## Check if this script is sourced
[[ "$0" != "${SOURCE_PATH}" ]] || ( echo "Vars script must be sourced." && exit 1) ;
## Determine location of this file and from this the location of the parent CleanStableDiffusion directory
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PARENT_PATH="$(realpath "${RELATIVE_PATH}/..")"
####################################

### User Configuration
export ENV_NAME="$(basename "$ABSOLUTE_PARENT_PATH")" # Default Name of the venv is the directory that contains this file
export ENV_DIR="${ABSOLUTE_PARENT_PATH}"/venv         # Default location of this VENV is "./venv"