#! /bin/bash

docker rmi $(docker images -a --filter=dangling=true -q)
docker buildx prune