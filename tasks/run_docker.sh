#! /bin/bash

source tasks/source_env.sh project.env


echo "$(docker ps -q -f name=$DOCKER_NAME)"
echo "$DOCKER_NAME"

# check if container is already running
if [ "$(docker ps -q -f name=$DOCKER_NAME)" ]; then
    echo "Container $DOCKER_NAME is already running. Please stop it first."
    exit 1
fi

./docker/run.sh \
    --name $DOCKER_NAME \
    --image $DOCKER_NAMESPACE/$DOCKER_NAME:$DOCKER_TAG