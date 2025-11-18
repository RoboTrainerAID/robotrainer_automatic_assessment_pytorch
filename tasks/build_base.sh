#! /bin/bash


source tasks/source_env.sh project.env

NAMESPACE=${1:-$DOCKER_NAMESPACE}
TAG=${2:-$DOCKER_TAG}

if [ "$DOCKER_BASE_CACHE" = "true" ]; then
    echo "Building base container with cache..."
    ./docker/build.sh \
        -b "BASE_IMAGE=$BASE_IMAGE TORCH_CUDA_ARCH=$TORCH_CUDA_ARCH MAX_BUILD_JOBS=$MAX_BUILD_JOBS" \
        -f ./docker/Dockerfile.base \
        -t $NAMESPACE/$DOCKER_NAME:$TAG-base
else
    echo "Building base container without cache..."
    ./docker/build.sh \
        -n \
        -b "BASE_IMAGE=$BASE_IMAGE TORCH_CUDA_ARCH=$TORCH_CUDA_ARCH MAX_BUILD_JOBS=$MAX_BUILD_JOBS" \
        -f ./docker/Dockerfile.base \
        -t $NAMESPACE/$DOCKER_NAME:$TAG-base
fi