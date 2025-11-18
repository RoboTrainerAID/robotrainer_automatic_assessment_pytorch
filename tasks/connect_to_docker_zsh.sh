#! /bin/bash

source tasks/source_env.sh project.env

docker exec -it $DOCKER_NAMESPACE/$DOCKER_NAME:$DOCKER_TAG /bin/zsh