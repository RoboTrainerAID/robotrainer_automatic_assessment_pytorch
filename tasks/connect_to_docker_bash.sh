#! /bin/bash

source tasks/source_env.sh project.env

docker exec -it $DOCKER_NAME /bin/bash