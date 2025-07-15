version_file="assets/version.txt"
init_file="torchtitan/__init__.py"
if [[ -n "$BUILD_VERSION" ]]; then
    # Update the version in version.txt
    echo "$BUILD_VERSION" > "$version_file"
    # Create a variable named __version__ at the end of __init__.py
    echo "__version__ = \"$BUILD_VERSION\"" >> "$init_file"
else
    echo "Error: BUILD_VERSION environment variable is not set or empty."
    exit 1
fi
