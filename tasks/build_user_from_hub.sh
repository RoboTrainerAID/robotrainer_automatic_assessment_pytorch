#! /bin/bash


source tasks/source_env.sh project.env


if $DOCKER_USER_CACHE; then
    NO_CACHE_FLAG=""
    echo "Building "$DOCKER_NAME" container with cache..."
else
    NO_CACHE_FLAG="-n"
    echo "Building "$DOCKER_NAME" container without cache..."
fi

if $DOCKER_DEBUG; then
    DEBUG="-d"
    echo "Building "$DOCKER_NAME" container with verbose information"
else
    DEBUG=""
fi

./docker/build.sh \
    $NO_CACHE_FLAG \
    $DEBUG \
    -b "USERIMAGE=$DOCKER_USERNAME/$DOCKER_HUB_REPO:$DOCKER_NAME-$DOCKER_HUB_TAG-base $DOCKER_USER_ARGS TORCH_CUDA_ARCH=$TORCH_CUDA_ARCH MAX_BUILD_JOBS=$MAX_BUILD_JOBS" \
    -f ./docker/Dockerfile \
    -t $DOCKER_NAMESPACE/$DOCKER_NAME:$DOCKER_TAG