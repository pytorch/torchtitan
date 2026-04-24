#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
ABSOLUTE_PARENT_PATH="$(realpath "${RELATIVE_PATH}/..")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/JSC_modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

python3 -m pip install --upgrade -r "${ABSOLUTE_PATH}"/JSC_requirements.txt

echo ""
echo "Created the virtual environment $ENV_NAME, to activate it use 'source $ABSOLUTE_PATH/activate.sh' "

echo ""
if ! source "${ABSOLUTE_PATH}/load_git_modules.sh"; then
    echo "Warning: git submodules could not be initialized."
    echo "You may need to run:"
    echo "    git submodule update --init --recursive"
    echo "manually"
fi