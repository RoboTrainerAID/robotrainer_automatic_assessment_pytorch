#! /bin/bash

source tasks/source_env.sh project.env
source tasks/source_env.sh setup/cli_color_scheme.env

./setup/install_gum.sh

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
GUM_BIN="$SCRIPT_DIR/../setup/.vendor/gum/bin/gum"

# Get Docker Hub credentials
HUB_USERNAME="$("$GUM_BIN" input --placeholder "Docker Hub Username")"
HUB_REPO="$("$GUM_BIN" input --placeholder "Docker Hub Repository (e.g. my-repo)")"
HUB_TAG="$("$GUM_BIN" input --placeholder "Docker Hub Tag (e.g. latest)")"
HUB_PW="$("$GUM_BIN" input --placeholder "Docker Hub password (e.g. 1234 :P)")"

./setup/set_env.sh DOCKER_USERNAME "$HUB_USERNAME" project.env
./setup/set_env.sh DOCKER_HUB_REPO "$HUB_REPO" project.env
./setup/set_env.sh DOCKER_HUB_TAG "$HUB_TAG" project.env

./tasks/build_base.sh 
docker tag $DOCKER_NAMESPACE/$DOCKER_NAME:$DOCKER_TAG-base $HUB_USERNAME/$HUB_REPO:$DOCKER_NAME-$HUB_TAG-base


docker login -u $HUB_USERNAME -p $HUB_PW

docker push $HUB_USERNAME/$HUB_REPO:$DOCKER_NAME-$HUB_TAG-base
