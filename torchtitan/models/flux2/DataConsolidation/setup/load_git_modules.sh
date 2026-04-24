#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
ABSOLUTE_PARENT_PATH="$(realpath "${RELATIVE_PATH}/..")"

echo "Making sure all git submodules are loaded..."
# Find the Git repository root even if run from anywhere
GIT_ROOT="$(git -C "$ABSOLUTE_PARENT_PATH" rev-parse --show-toplevel 2>/dev/null)"

if [ -z "$GIT_ROOT" ]; then
    echo ""
    echo "Error: Could not find the Git repository root."
    echo "Make sure you run this inside the repository or clone it properly."
    echo ""
    return 1
fi

# Update submodules
if git -C "$GIT_ROOT" submodule update --init --recursive; then
    echo "... all git submodules initialised"
else
    echo " Failed to initialize git submodules!"
    return 1
fi
