#! /bin/bash

source tasks/source_env.sh setup/cli_color_scheme.env


SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
GUM_BIN="$SCRIPT_DIR/setup/.vendor/gum/bin/gum"

"$GUM_BIN" confirm "Do you really want to remove all files necessary for setting up this workspace?" && echo "removing /setup directory as well as install.sh and clean_setup_files.sh"