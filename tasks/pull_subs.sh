#! /bin/bash

# Find all directories in project that contain a .git directory (actual git repos)
# Exclude common non-git directories
PROJECT_FOLDERS=$(find project -maxdepth 1 -type d ! -path "project" ! -path "project/.git" ! -path "project/.vscode" ! -path "project/__pycache__" ! -path "project/node_modules")

if [ -z "$PROJECT_FOLDERS" ]; then
    echo "No project folders found in 'project' directory."
    exit 1
fi

for FOLDER in $PROJECT_FOLDERS; do
    # Only process if it's actually a git repository
    if [ -d "$FOLDER/.git" ]; then
        echo "Pulling changes in $FOLDER"
        if git -C "$FOLDER" pull; then
            echo "✓ Successfully pulled $FOLDER"
        else
            echo "✗ Failed to pull $FOLDER"
        fi
    else
        echo "⚠ Skipping $FOLDER (not a git repository)"
    fi
done