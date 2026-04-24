#!/bin/bash

# Function to recursively touch files
touch_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            touch_files "$file"
        else
            touch "$file"
        fi
    done
}

# Call the function with the target directory
touch_files "/p/scratch/nxtaim-1/proprietary/dspace"
