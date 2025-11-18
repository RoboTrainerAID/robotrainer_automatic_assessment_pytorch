#!/bin/bash

# Script to parse project.repos using a temporary Python virtual environment

# Configuration
VENV_DIR=".temp_venv"
PYTHON_SCRIPT="tasks/parse_repos.py"
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
GUM_BIN="$SCRIPT_DIR/../setup/.vendor/gum/bin/gum"

source tasks/source_env.sh setup/cli_color_scheme.env

    
# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi


python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet pyyaml

# parsing
python3 "$PYTHON_SCRIPT" project.repos > .env_export
source .env_export

deactivate
rm -f .env_export
rm -rf "$VENV_DIR"

# Find all FOLDER_ variables and iterate through them
i=0
while true; do
    folder_var="FOLDER_$i"
    url_var="URL_$i"
    version_var="VERSION_$i"
    
    # Use indirect variable expansion to get the value
    folder_value="${!folder_var}"
    url_value="${!url_var}"
    version_value="${!version_var}"
    
    # Break if the variable doesn't exist
    if [[ -z "$folder_value" ]]; then
        break
    fi

    if [ ! -d "project/$folder_value" ]; then
        git clone --branch "$version_value" "$url_value" "project/$folder_value"
    else
        "$GUM_BIN" style "Directory project/$("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "$folder_value") already exists. Skipping clone."

    fi

    ((i++))
done