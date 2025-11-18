#! /bin/bash

# Install script for setting up the environment
# Created by Philipp Augenstein as part of the IRAS Perception Group
# Contact: Philipp.Augenstein@h-ka.de
# Last updated: 2025-08-25

# Tested on Ubuntu 22.04, 22.04
# Requires: git, vscode
# Please report any issues to the contact above

SNIPPET_DIR="setup/snippets/"

source tasks/source_env.sh setup/cli_color_scheme.env
source tasks/source_env.sh project.env

./setup/install_gum.sh

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
GUM_BIN="$SCRIPT_DIR/setup/.vendor/gum/bin/gum"

"$GUM_BIN" style --border normal --margin "2" --padding "1 2" --border-foreground "$ACCENT_COLOR_4" "This is the IRAS Developing in Docker Workspace Setup Script.
This system needs to have the following dependencies installed:

$("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "python3    git    vscode    docker    GPU driver (for CUDA)")"

# check if python is installed 
if ! command -v python3 &> /dev/null
then
    if ! command -v python &> /dev/null
    then
        echo "Python could not be found, please install Python to proceed."
        exit
    else
        alias python3=python
    fi
fi

python3 ./setup/install_vscode_extensions.py

# query for project information 
BASE_CONTAINER_TYPE="$("$GUM_BIN" choose --header "Select the base container type" \
    "CUDA" \
    "ROS" \
    "Pytorch" \
    "Pytorch built from source" \
    "Python Datascience")"

# ROS was chosen
if [ "$BASE_CONTAINER_TYPE" == "ROS" ]; then
    ROS_DISTRO="$("$GUM_BIN" input --placeholder "Which distribution do you like? Please enter the image tag.")"
    BASE_IMAGE="ros:$ROS_DISTRO"
    python3 ./setup/recommend_project_extensions.py ${SNIPPET_DIR}ros/extensions.json.ros
    ./setup/set_env.sh BASE_IMAGE "$BASE_IMAGE" project.env
    cp ${SNIPPET_DIR}ros/Dockerfile.ROS docker/Dockerfile.base
    cp ${SNIPPET_DIR}ros/dds_profile.xml docker/dds_profile.xml

    ./tasks/stitch_md.sh -s -t "ROS Specifics" README.md setup/snippets/readme.ros.md
    
    "$GUM_BIN" style --border normal --margin "2" --padding "1 2" --border-foreground "$ACCENT_COLOR_4" "You have chosen the ROS base image.
The Dockerfile has been set to $("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "Dockerfile.ROS")
The base image has been set to $("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "$BASE_IMAGE")
The ROS Readme file has been attached to the main Readme.
You can change these settings later in the project.env file"
fi

if [ "$BASE_CONTAINER_TYPE" == "CUDA" ] || [ "$BASE_CONTAINER_TYPE" == "Pytorch" ] || [ "$BASE_CONTAINER_TYPE" == "Pytorch built from source" ]; then
    "$GUM_BIN" confirm "Do you want the autodetected CUDA version from the current system?" && {
        "$GUM_BIN" spin --title "Looking for the $("$GUM_BIN" style --foreground "$ACCENT_COLOR_1" "best CUDA version.")" -- sleep 1.8
        # get cuda version from host using nvidia-smi
        if command -v nvidia-smi &> /dev/null; then
            CUDA_HOST_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
            if [ -n "$CUDA_HOST_VERSION" ]; then
                echo "Detected CUDA version: $CUDA_HOST_VERSION"
            else
                echo "Could not detect CUDA version from nvidia-smi output"
            fi
        else
            echo "nvidia-smi not found. CUDA may not be installed or available."
        fi
        TORCH_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
    }
    if [ -z "$CUDA_HOST_VERSION" ]; then
        CUDA_HOST_VERSION=$("$GUM_BIN" input --placeholder "Please input the CUDA version you want to use (e.g., 11.8, 12.8, ...)")
        TORCH_CUDA_ARCH=$("$GUM_BIN" input --placeholder "Please input the CUDA compute capability of your GPU (e.g., 7.5, 8.6, ...)")
    fi

    ./setup/set_env.sh TORCH_CUDA_ARCH "$TORCH_CUDA_ARCH" project.env
    if [ "$BASE_CONTAINER_TYPE" == "Pytorch" ]; then
        case $CUDA_HOST_VERSION in
            "11.8")
                BASE_IMAGE="pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel"
                ;;
            "12.1")
                BASE_IMAGE="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"
                ;;
            "12.4")
                BASE_IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
                ;;
            "12.6")
                BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel"
                ;;
            "12.8")
                BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
                ;;
            "12.9")
                BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel"
                ;;    
            *)
                echo "Unsupported CUDA version for prebuilt PyTorch image. Supported versions are 11.8, 12.1, 12.4, 12.6, 12.8, 12.9"
                exit 1
                ;;
        esac
        cp ${SNIPPET_DIR}pytorch/Dockerfile.Pytorch docker/Dockerfile.base
        "$GUM_BIN" style --border normal --margin "2" --padding "1 2" --border-foreground "$ACCENT_COLOR_4" "You have chosen the PyTorch base image.
The Base image has been set to $("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "$BASE_IMAGE")."
    fi

    if [ "$BASE_CONTAINER_TYPE" == "Pytorch built from source" ]; then
        cp ${SNIPPET_DIR}pytorch/Dockerfile.PytorchFromSource docker/Dockerfile.base
        case $CUDA_HOST_VERSION in
            "11.8")
                BASE_IMAGE="11.8.0"
                ;;
            "12.1")
                BASE_IMAGE="12.1.0"
                ;;
            "12.4")
                BASE_IMAGE="12.4.0"
                ;;
            "12.6")
                BASE_IMAGE="12.6.0"
                ;;
            "12.8")
                BASE_IMAGE="12.8.1"
                ;;
            "12.9")
                BASE_IMAGE="12.9.0"
                ;;    
            *)
                echo "Unsupported CUDA version for building PyTorch from source. Supported versions are 11.8, 12.1, 12.4, 12.6, 12.8, 12.9"
                exit 1
                ;;
        esac
        "$GUM_BIN" style --border normal --margin "2" --padding "1 2" --border-foreground "$ACCENT_COLOR_4" "You have chosen to build PyTorch from source.
The Base image has been set to $("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "nvidia/cuda:$BASE_IMAGE-cudnn-devel-ubuntu22.04")."
    fi
    ./setup/set_env.sh BASE_IMAGE "$BASE_IMAGE" project.env

fi

## Get project information
DOCKER_NAME="$("$GUM_BIN" input --placeholder "Please enter a name for your project (image+container).")"
DOCKER_TAG="$("$GUM_BIN" input --header "Docker Tag" --placeholder "Please enter a tag for your project docker." --value "latest")"
DOCKER_NAMESPACE="$("$GUM_BIN" input --header "Docker Namespace" --placeholder "Please enter a namespace for your project." --value "iras")"

./setup/set_env.sh DOCKER_NAME "$DOCKER_NAME" project.env
./setup/set_env.sh DOCKER_TAG "$DOCKER_TAG" project.env
./setup/set_env.sh DOCKER_NAMESPACE "$DOCKER_NAMESPACE" project.env

## Clone Subrepos
./tasks/clone_from_file.sh

# Get the list of repositories for display
REPO_LIST=$(cat project.repos | grep -v '^#' | awk '{print $2}' | tr '\n' ' ')

"$GUM_BIN" style --border normal --margin "2" --padding "1 2" --border-foreground "$ACCENT_COLOR_4" "Your system has been set up. The following subrepositories have been cloned:

$("$GUM_BIN" style --foreground "$ACCENT_COLOR_3" --bold "$REPO_LIST")

You can now use the buttons in the bottom left corner of VSCode to build and run your container.
Happy coding!"