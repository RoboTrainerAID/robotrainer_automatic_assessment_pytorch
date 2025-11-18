#! /bin/bash

source tasks/source_env.sh project.env

docker stop $DOCKER_NAME
