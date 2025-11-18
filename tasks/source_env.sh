#! /bin/bash

# Source the file to expand variables, then export them
set -a  # automatically export all variables
source "$1"
set +a  # turn off automatic export