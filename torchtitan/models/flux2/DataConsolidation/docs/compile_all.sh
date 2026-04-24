#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR" || exit

rm -rf "$SCRIPT_DIR/build/"* "$SCRIPT_DIR/source/generated/"*

# Ensure the image symlink exists
IMG_SOURCE_DIR="$SCRIPT_DIR/../imgs"
IMG_LINK_TARGET="$SCRIPT_DIR/source/imgs"

# Remove existing broken or outdated symlink
if [ -L "$IMG_LINK_TARGET" ] || [ -e "$IMG_LINK_TARGET" ]; then
    rm -rf "$IMG_LINK_TARGET"
fi

ln -s "$IMG_SOURCE_DIR" "$IMG_LINK_TARGET"

# Check for required Python packages
REQUIRED_MODULES=(sphinx sphinx_rtd_theme)
MISSING_MODULES=()

for module in "${REQUIRED_MODULES[@]}"; do
    python3 -c "import $module" 2>/dev/null || MISSING_MODULES+=("$module")
done

if [ ${#MISSING_MODULES[@]} -ne 0 ]; then
    echo "Error: Missing Python modules: ${MISSING_MODULES[*]}"
    echo "You can install them by running:"
    echo "  pip install ${MISSING_MODULES[*]}"
    exit 1
fi

make html

python3 -m http.server 8000 --directory "$SCRIPT_DIR/build/html"
