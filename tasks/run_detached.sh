#! /bin/bash

source tasks/source_env.sh project.env

./docker/run.sh \
    --detach \
    --name $DOCKER_NAME \
    --image $DOCKER_NAMESPACE/$DOCKER_NAME:$DOCKER_TAG-dev