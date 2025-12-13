init_file="torchtitan/__init__.py"
pyproject_file="pyproject.toml"
if [[ -n "$BUILD_VERSION" ]]; then
    # Create a variable named __version__ at the end of __init__.py
    echo "__version__ = \"$BUILD_VERSION\"" >> "$init_file"
    # Replace dynamic version with fixed version in pyproject.toml
    sed -i '' 's/dynamic = \["version"\]/version = "'"$BUILD_VERSION"'"/' "$pyproject_file"
else
    echo "Error: BUILD_VERSION environment variable is not set or empty."
    exit 1
fi
