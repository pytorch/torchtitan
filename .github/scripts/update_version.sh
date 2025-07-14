python_file="assets/version.txt"
if [[ -n "$BUILD_VERSION" ]]; then
    echo "$BUILD_VERSION" > "$python_file"
else
    echo "Error: BUILD_VERSION environment variable is not set or empty."
    exit 1
fi
