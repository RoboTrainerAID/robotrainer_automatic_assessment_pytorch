#! /bin/bash

CACHE=""
TAG="latest"
DEBUG=""
FILE="./docker/Dockerfile"
BUILD_ARGS=""

# Parse command-line arguments
while getopts ":hndf:t:b:" opt; do
    case $opt in
        h)
            echo "Build docker container for development."
            echo "  -h        Show help"
            echo "  -n        Do not use cache when building the container"
            echo "  -d        Build for debug information (verbose build process)"
            echo "  -f FILE   Specify the Dockerfile to use"
            echo "  -t TAG    Specify the image tag"
            echo "  -b ARGS   Pass multiple build arguments to docker build (e.g., -b 'TORCH=\"abc\" USERARG_1=\"cde\"')"
            exit 0
            ;;
        n)
            CACHE="--no-cache"
            ;;
        d)
            DEBUG="--progress=plain"
            ;;
        f)
            FILE="$OPTARG"
            ;;
        t)
            TAG="$OPTARG"
            ;;
        b)
            # Parse multiple build arguments from a single -b option
            # Split OPTARG by spaces and handle quoted values
            IFS=' ' read -ra ARGS <<< "$OPTARG"
            for arg in "${ARGS[@]}"; do
                BUILD_ARGS="$BUILD_ARGS --build-arg $arg"
            done
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if [[ $CACHE == "--no-cache" ]]; then
    echo "Building container without cache..."
fi


DOCKER_BUILDKIT=1 docker buildx build $CACHE $DEBUG --output type=image,compression=zstd $BUILD_ARGS -f $FILE -t $TAG .

#echo "$CACHE $DEBUG $BUILD_ARGS $FILE $TAG"
